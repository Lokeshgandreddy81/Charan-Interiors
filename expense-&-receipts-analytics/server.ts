import "dotenv/config";
import express from "express";
import fs from "fs";
import path from "path";
import { fileURLToPath } from "url";
import {
  GoogleGenAI,
  Type,
  type GenerateContentParameters,
} from "@google/genai";
import { execFile } from "child_process";
import { promisify } from "util";
import os from "os";
import { createServer as createViteServer } from "vite";
import {
  normalizeChapterNumber,
  sortChapterItems,
} from "./src/utils/chapterSorting.ts";
import { PDFDocument } from "pdf-lib";

const execFileAsync = promisify(execFile);

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

export const MODEL_ID = "gemini-3.8-flash";
const FALLBACK_MODELS = [
  "gemini-flash-latest",
  "gemini-3.5-flash",
  "gemini-3.1-pro-preview",
  "gemini-2.5-flash",
];
let activeModelId: string = MODEL_ID;
const unavailableModels = new Set<string>();
const LOG_FILE = "/tmp/shoebox.log";

function withTimeout<T>(
  promise: Promise<T>,
  timeoutMs: number,
  errMsg: string
): Promise<T> {
  let timer: NodeJS.Timeout;
  const timeoutPromise = new Promise<T>((_, reject) => {
    timer = setTimeout(() => reject(new Error(errMsg)), timeoutMs);
  });
  return Promise.race([promise, timeoutPromise]).finally(() => {
    clearTimeout(timer);
  });
}

function isTransientError(msg: string): boolean {
  const lower = msg.toLowerCase();
  return (
    lower.includes("503") ||
    lower.includes("unavailable") ||
    lower.includes("high demand") ||
    lower.includes("spikes in demand") ||
    lower.includes("overloaded") ||
    lower.includes("temporarily unavailable") ||
    lower.includes("try again later") ||
    lower.includes("429") ||
    lower.includes("resource_exhausted") ||
    lower.includes("quota") ||
    lower.includes("rate limit") ||
    lower.includes("500") ||
    lower.includes("internal") ||
    lower.includes("502") ||
    lower.includes("bad gateway") ||
    lower.includes("504") ||
    lower.includes("gateway timeout") ||
    lower.includes("fetch failed") ||
    lower.includes("econnreset") ||
    lower.includes("etimedout") ||
    lower.includes("timed out")
  );
}

function isQuotaExhaustedError(msg: string): boolean {
  const lower = msg.toLowerCase();
  return (
    lower.includes("resource_exhausted") ||
    lower.includes("quota exceeded") ||
    lower.includes("exceeded your current quota") ||
    lower.includes("free tier") ||
    lower.includes("credit balance")
  );
}

function isModelCascadeError(msg: string): boolean {
  const lower = msg.toLowerCase();
  return (
    isTransientError(msg) ||
    lower.includes("404") ||
    lower.includes("not found") ||
    lower.includes("not supported") ||
    lower.includes("no longer available")
  );
}

function extractHumanErrorMessage(err: unknown, fallback: string): string {
  if (!err) return fallback;
  const rawMsg = err instanceof Error ? err.message : String(err);
  try {
    const parsed = JSON.parse(rawMsg);
    if (parsed?.error?.message) {
      return parsed.error.message;
    }
  } catch {
    // not JSON
  }
  return rawMsg;
}

function logEvent(message: string, meta?: Record<string, unknown>) {
  const line = `[${new Date().toISOString()}] ${message}${
    meta ? " " + JSON.stringify(meta) : ""
  }\n`;
  process.stdout.write(line);
  fs.appendFile(LOG_FILE, line, "utf8", () => {});
}

function parseJsonSafely<T>(rawText: string | undefined, fallback: T): T {
  if (!rawText) return fallback;
  let cleaned = rawText.trim();
  if (cleaned.startsWith("```")) {
    cleaned = cleaned
      .replace(/^```(?:json)?\s*/i, "")
      .replace(/\s*```$/, "")
      .trim();
  }
  try {
    return JSON.parse(cleaned) as T;
  } catch {
    const firstBrace = cleaned.indexOf("{");
    const lastBrace = cleaned.lastIndexOf("}");
    if (firstBrace !== -1 && lastBrace > firstBrace) {
      try {
        return JSON.parse(cleaned.slice(firstBrace, lastBrace + 1)) as T;
      } catch {
        // continue
      }
    }
    const firstBracket = cleaned.indexOf("[");
    const lastBracket = cleaned.lastIndexOf("]");
    if (firstBracket !== -1 && lastBracket > firstBracket) {
      try {
        return JSON.parse(cleaned.slice(firstBracket, lastBracket + 1)) as T;
      } catch {
        // continue
      }
    }
    return fallback;
  }
}

const PLACEHOLDER_API_KEYS = new Set([
  "",
  "MY_GEMINI_API_KEY",
  "PLACEHOLDER_API_KEY",
  "YOUR_GEMINI_API_KEY",
  "YOUR_API_KEY",
  "TODO",
  "undefined",
  "null",
]);

function isValidApiKey(val: unknown): val is string {
  if (typeof val !== "string") return false;
  const trimmed = val.trim();
  return trimmed.length > 0 && !PLACEHOLDER_API_KEYS.has(trimmed);
}

function readDevEnvJson(): Record<string, unknown> {
  const candidates = [
    path.join(process.cwd(), ".dev.env.json"),
    "/tmp/.dev.env.json",
  ];
  for (const filePath of candidates) {
    try {
      if (fs.existsSync(filePath)) {
        const raw = fs.readFileSync(filePath, "utf8");
        const parsed = JSON.parse(raw);
        if (parsed && typeof parsed === "object") {
          return parsed as Record<string, unknown>;
        }
      }
    } catch {
      // ignore unreadable or malformed .dev.env.json
    }
  }
  return {};
}

function resolveApiKey(): string | undefined {
  for (const candidate of [process.env.GEMINI_API_KEY, process.env.API_KEY]) {
    if (isValidApiKey(candidate)) {
      return candidate.trim();
    }
  }

  const devEnv = readDevEnvJson();
  for (const candidate of [devEnv.GEMINI_API_KEY, devEnv.API_KEY]) {
    if (isValidApiKey(candidate)) {
      return candidate.trim();
    }
  }

  return undefined;
}

let cachedClient: GoogleGenAI | null = null;
let cachedClientKey: string | undefined = undefined;

function getGenAIClient(): GoogleGenAI {
  const apiKey = resolveApiKey();
  if (!apiKey) {
    throw new Error(
      "GEMINI_API_KEY environment variable is missing or set to a placeholder value."
    );
  }
  if (cachedClient && cachedClientKey === apiKey) {
    return cachedClient;
  }
  cachedClient = new GoogleGenAI({
    apiKey,
    httpOptions: {
      headers: {
        "User-Agent": "aistudio-build",
      },
    },
  });
  cachedClientKey = apiKey;
  return cachedClient;
}

async function generateWithModelFallback(
  params: Omit<GenerateContentParameters, "model">
): Promise<{
  response: Awaited<ReturnType<GoogleGenAI["models"]["generateContent"]>>;
  modelUsed: string;
}> {
  const ai = getGenAIClient();
  const allOrdered = [
    activeModelId,
    ...FALLBACK_MODELS.filter((m) => m !== activeModelId),
  ];
  const candidates = allOrdered.filter((m) => !unavailableModels.has(m));
  const pool = candidates.length > 0 ? candidates : allOrdered;

  let lastErr: unknown = null;
  for (const candidateModel of pool) {
    // Attempt up to 2 tries per model with backoff on transient errors (503 / 429)
    for (let attempt = 1; attempt <= 2; attempt++) {
      try {
        logEvent(
          `Calling Gemini model ${candidateModel} (attempt ${attempt})...`
        );
        const response = await withTimeout(
          ai.models.generateContent({
            ...params,
            model: candidateModel,
          }),
          90000,
          `Model ${candidateModel} timed out after 90s`
        );
        activeModelId = candidateModel;
        unavailableModels.delete(candidateModel);
        return { response, modelUsed: candidateModel };
      } catch (err: unknown) {
        lastErr = err;
        const rawMsg = err instanceof Error ? err.message : String(err);
        const isTransient = isTransientError(rawMsg);
        const isCascade = isModelCascadeError(rawMsg);

        const causeMsg =
          (err as any)?.cause?.message || (err as any)?.cause
            ? ` [cause: ${(err as any)?.cause?.message || (err as any)?.cause}]`
            : "";
        logEvent(
          `Model ${candidateModel} attempt ${attempt} failed: ${rawMsg}${causeMsg}`
        );

        if (!isCascade) {
          throw err;
        }

        // If the account quota is completely exhausted, do not retry or cascade through all other models
        if (isQuotaExhaustedError(rawMsg)) {
          unavailableModels.add(candidateModel);
          logEvent(
            `Gemini API quota exhausted (${rawMsg}). Halting model retries to allow instant local fallback.`
          );
          throw err;
        }

        // If the model is fundamentally unavailable or unsupported (404), cascade immediately
        if (!isTransient && isCascade) {
          unavailableModels.add(candidateModel);
          logEvent(
            `Model ${candidateModel} permanently unavailable/unsupported, cascading immediately...`
          );
          break;
        }

        if (attempt === 1) {
          // Jittered backoff (1200-1600ms) for temporary demand spikes before retrying same model
          const delay = 1200 + Math.floor(Math.random() * 400);
          await new Promise((resolve) => setTimeout(resolve, delay));
          continue;
        }

        // Exhausted attempts on this model, mark temporarily unavailable and cascade to next model in pool
        unavailableModels.add(candidateModel);
        logEvent(
          `Model ${candidateModel} capacity exhausted, cascading to next fallback model...`
        );
        break;
      }
    }
  }

  // If all models in pool failed, clear unavailable cache and throw clean error
  unavailableModels.clear();
  throw lastErr;
}

const READ_SYSTEM_INSTRUCTION = `You are a careful, meticulous bookkeeper examining photos of physical receipts and business invoices. Your job is to find every receipt in every photo, box each one accurately, and extract its printed and handwritten values.

What to look for and catch that a basic scanner misses:
- Several receipts per photo: Box each physical receipt separately using box_2d = [ymin, xmin, ymax, xmax] normalised 0–1000 relative to the photo, even when receipts overlap or are tilted.
- Which number is the total: Find the true total charged, NOT the subtotal, NOT "cash tendered", and NOT "change". When there is no handwritten tip, set both printed_total and final_total to the same true total charged.
- Handwritten tips on card slips that change the printed total: Return printed_total, handwritten_tip, and final_total separately. If there is a handwritten tip that changes the total, add a flag with type "tip_changes_total". If there is no handwritten tip on the receipt, simply return null for handwritten_tip with NO flag.
- Foreign currency: Read the ISO 3-letter currency code (e.g., USD, GBP, EUR, NZD, AUD, JPY, CAD) from currency symbols, words, tax labels, or shop address. Never convert currencies.
- Tax lines (VAT, GST, consumption tax): Read the tax amount and tax_label (e.g. "VAT 20%", "GST 15%"). If a tax line is visibly printed on the receipt but blurred/unreadable, return null for tax, lower tax confidence, and add a flag with type "unreadable_field" and note "tax unreadable". If the receipt simply has no tax line, return null for tax and tax_label with NO flag.
- B2B Tax & Invoice compliance fields:
  - tax_id_or_vat: Extract the merchant's VAT, GST, EIN, or ABN tax registration ID if printed on the receipt (e.g. "GB 123 4567 89", "GST 12-345-678"), or null if not printed.
  - is_tax_invoice: Set to true if the receipt shows an itemized tax/VAT line, a merchant tax registration number, or a formal "TAX INVOICE" header; set to false if it is only a summary card terminal slip or receipt without tax details.
- Faded, creased, stained, or cut-off text: Only add an "unreadable_field" or "low_confidence" flag when a core printed field (merchant, date, total, or a visibly present tax/tip line) is damaged, blurry, or cut off. Do NOT add flags for optional fields (like handwritten_tip, time, item_count, card_last4, or tax_id_or_vat) that are simply not printed on the receipt.
- Time and item count: Read "time" (as HH:MM:SS or HH:MM as printed, or null if not printed) and "item_count" (explicit printed item count or count of printed line items on the receipt, or null if not printed or unreadable). Never guess either field.

Strict rules:
1. Return one entry per physical receipt, even if the same receipt appears in two photos (a separate duplicate pass handles that).
2. Every total, tax, and tip must be read directly from the receipt. Never compute or estimate a missing monetary value; return null and a flag.
3. Categories must come strictly from this fixed list: "Meals", "Travel", "Lodging", "Office", "Other".
4. Never give advice on what is tax-deductible, allowable, or claimable.
5. Date should be formatted as YYYY-MM-DD when possible, or as printed if partial, or null if unreadable.`;

const READ_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    receipts: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          photo_index: {
            type: Type.NUMBER,
            description: "The 1-based global photo index passed in the prompt.",
          },
          box_2d: {
            type: Type.ARRAY,
            items: { type: Type.NUMBER },
            description: "Bounding box [ymin, xmin, ymax, xmax] normalised 0 to 1000.",
          },
          merchant: {
            type: Type.STRING,
            nullable: true,
            description: "Merchant or shop name as printed on the receipt, or null if unreadable.",
          },
          date: {
            type: Type.STRING,
            nullable: true,
            description: "Receipt date in YYYY-MM-DD format if possible, or null if unreadable.",
          },
          time: {
            type: Type.STRING,
            nullable: true,
            description: "Time printed on receipt (e.g. '09:34:12' or '09:34'), or null if not present.",
          },
          item_count: {
            type: Type.NUMBER,
            nullable: true,
            description: "Number of items read from the receipt, or null if not present.",
          },
          printed_total: {
            type: Type.NUMBER,
            nullable: true,
            description: "Printed total before handwritten tip, or null if unreadable.",
          },
          handwritten_tip: {
            type: Type.NUMBER,
            nullable: true,
            description: "Handwritten tip or gratuity amount if present, otherwise null.",
          },
          final_total: {
            type: Type.NUMBER,
            nullable: true,
            description: "Final total written or printed on the receipt, or null if unreadable.",
          },
          currency: {
            type: Type.STRING,
            nullable: true,
            description: "3-letter currency code read from symbols, words, or address (e.g. USD, GBP, JPY, NZD, EUR).",
          },
          tax: {
            type: Type.NUMBER,
            nullable: true,
            description: "Tax amount printed on the receipt, or null if not shown or unreadable.",
          },
          tax_label: {
            type: Type.STRING,
            nullable: true,
            description: "Tax label printed on the receipt (e.g. 'VAT 20%', 'GST'), or null.",
          },
          tax_id_or_vat: {
            type: Type.STRING,
            nullable: true,
            description: "Merchant VAT, GST, EIN, or ABN tax registration ID if printed on the receipt, or null.",
          },
          is_tax_invoice: {
            type: Type.BOOLEAN,
            nullable: true,
            description: "True if the receipt shows an itemized tax/VAT line, tax registration number, or formal TAX INVOICE header; false if only a card slip or receipt without tax details.",
          },
          category: {
            type: Type.STRING,
            description: "Fixed category: Meals, Travel, Lodging, Office, or Other.",
          },
          card_last4: {
            type: Type.STRING,
            nullable: true,
            description: "Last 4 digits of payment card if shown (e.g. '0000'), or null.",
          },
          confidence: {
            type: Type.OBJECT,
            properties: {
              merchant: { type: Type.NUMBER },
              date: { type: Type.NUMBER },
              printed_total: { type: Type.NUMBER },
              handwritten_tip: { type: Type.NUMBER },
              currency: { type: Type.NUMBER },
              tax: { type: Type.NUMBER },
            },
            required: ["merchant", "date", "printed_total", "handwritten_tip", "currency", "tax"],
          },
          flags: {
            type: Type.ARRAY,
            items: {
              type: Type.OBJECT,
              properties: {
                type: {
                  type: Type.STRING,
                  description: "One of: tip_changes_total, unreadable_field, foreign_currency, possible_duplicate, low_confidence, not_a_receipt",
                },
                note: {
                  type: Type.STRING,
                  description: "Short human-readable explanation of the flag.",
                },
              },
              required: ["type", "note"],
            },
          },
        },
        required: [
          "photo_index",
          "box_2d",
          "merchant",
          "date",
          "time",
          "item_count",
          "printed_total",
          "handwritten_tip",
          "final_total",
          "currency",
          "tax",
          "tax_label",
          "tax_id_or_vat",
          "is_tax_invoice",
          "category",
          "card_last4",
          "confidence",
          "flags",
        ],
      },
    },
  },
  required: ["receipts"],
};

const DUPLICATES_RESPONSE_SCHEMA = {
  type: Type.OBJECT,
  properties: {
    duplicates: {
      type: Type.ARRAY,
      items: {
        type: Type.OBJECT,
        properties: {
          row: {
            type: Type.NUMBER,
            description: "The later 1-based row number that is a likely duplicate.",
          },
          duplicate_of: {
            type: Type.NUMBER,
            description: "The earlier 1-based row number that this row duplicates.",
          },
          reason: {
            type: Type.STRING,
            description: "Clear explanation, e.g. 'Same merchant, date, total and card; photographed twice' or 'Cross-source duplicate between Google Drive and Gmail'.",
          },
        },
        required: ["row", "duplicate_of", "reason"],
      },
    },
  },
  required: ["duplicates"],
};

async function startServer() {
  const app = express();
  const PORT = Number(process.env.PORT) || 3000;

  // Support batches and large PDFs up to 100MB payload
  app.use(express.json({ limit: "100mb" }));
  app.use(express.urlencoded({ extended: true, limit: "100mb" }));

  app.get("/api/workspace-config", (_req, res) => {
    const devEnv = readDevEnvJson();
    let firebaseConfig: Record<string, unknown> | null = null;
    const configCandidates = [
      path.join(process.cwd(), "firebase-applet-config.json"),
      path.join(__dirname, "firebase-applet-config.json"),
    ];
    for (const candidatePath of configCandidates) {
      try {
        if (fs.existsSync(candidatePath)) {
          const parsed = JSON.parse(fs.readFileSync(candidatePath, "utf8"));
          if (parsed && typeof parsed === "object") {
            const apiKey =
              typeof parsed.apiKey === "string" ? parsed.apiKey.trim() : "";
            const authDomain =
              typeof parsed.authDomain === "string"
                ? parsed.authDomain.trim()
                : "";
            const projectId =
              typeof parsed.projectId === "string"
                ? parsed.projectId.trim()
                : "";
            const isValidFb =
              apiKey.length > 0 &&
              !PLACEHOLDER_API_KEYS.has(apiKey) &&
              !apiKey.includes("remixed-") &&
              authDomain.length > 0 &&
              !authDomain.includes("remixed-") &&
              !projectId.includes("remixed-");
            if (isValidFb) {
              firebaseConfig = parsed as Record<string, unknown>;
              break;
            }
          }
        }
      } catch {
        // ignore unreadable firebase-applet-config.json
      }
    }

    const rawFbClientId =
      (typeof firebaseConfig?.oAuthClientId === "string"
        ? firebaseConfig.oAuthClientId.trim()
        : null) ||
      (typeof firebaseConfig?.oauthClientId === "string"
        ? firebaseConfig.oauthClientId.trim()
        : null);
    const fbClientId =
      rawFbClientId &&
      !PLACEHOLDER_API_KEYS.has(rawFbClientId) &&
      !rawFbClientId.includes("remixed-")
        ? rawFbClientId
        : null;

    const rawId =
      process.env.GOOGLE_CLIENT_ID ||
      process.env.VITE_GOOGLE_CLIENT_ID ||
      (typeof devEnv.GOOGLE_CLIENT_ID === "string"
        ? devEnv.GOOGLE_CLIENT_ID
        : null) ||
      (typeof devEnv.VITE_GOOGLE_CLIENT_ID === "string"
        ? devEnv.VITE_GOOGLE_CLIENT_ID
        : null) ||
      fbClientId;

    const cleanId =
      rawId &&
      !PLACEHOLDER_API_KEYS.has(rawId.trim()) &&
      !rawId.trim().includes("remixed-")
        ? rawId.trim()
        : null;

    res.json({
      firebaseConfig,
      googleClientId: cleanId,
    });
  });

  app.get("/api/logs", (_req, res) => {
    try {
      const content = fs.existsSync(LOG_FILE)
        ? fs.readFileSync(LOG_FILE, "utf8")
        : "No logs yet.";
      res.type("text/plain").send(content);
    } catch (e) {
      res.status(500).send(String(e));
    }
  });

  app.post("/api/chat", async (req, res) => {
    const startTime = Date.now();
    try {
      const { messages, context, model } = req.body as {
        messages: Array<{ role: "user" | "assistant"; content: string }>;
        context?: {
          docTitle?: string;
          chapters?: Array<{ title: string; summary?: string; pageRange?: string }>;
        };
        model?: string;
      };

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        res.status(400).json({ error: "No messages provided." });
        return;
      }

      const lastMessage = messages[messages.length - 1];
      const systemInstruction = `You are Unify AI, an ultra-clean, decisive, and exceptionally sharp document intelligence assistant.
Your purpose is to assist the user with document synthesis, research, prospecting, and analysis.
Context:
${context?.docTitle ? `Active Document: "${context.docTitle}"` : ""}
${context?.chapters && context.chapters.length > 0 ? `Document Chapters Outline:\n${context.chapters.map((c, i) => `${i + 1}. ${c.title}${c.summary ? `: ${c.summary}` : ""}`).join("\n")}` : ""}

Style Guidelines:
- Respond cleanly, concisely, and insightfully.
- Keep the structure neat: use bold subheadings, clean bullet points, or concise numbered steps where helpful.
- Avoid unnecessary fluff or overly robotic pleasantries. Be direct, authoritative, and helpful.`;

      const contents = messages
        .map((m) => `${m.role === "user" ? "User" : "Assistant"}: ${m.content}`)
        .join("\n\n");

      const { response, modelUsed } = await generateWithModelFallback({
        contents: `${contents}\n\nAssistant:`,
        config: {
          systemInstruction,
          temperature: 0.2,
        },
      });

      const elapsedMs = Date.now() - startTime;
      const thinkingSeconds = Math.max(1, Math.round(elapsedMs / 1000));

      res.json({
        content: response.text || "No response received.",
        thinkingTime: `${thinkingSeconds}s`,
        stepTitle: "Narrowing target criteria",
        stepDescription:
          "Synthesizing signals and matching criteria across available sources.",
        modelUsed,
      });
    } catch (error: unknown) {
      const elapsedMs = Date.now() - startTime;
      const message =
        error instanceof Error ? error.message : "Failed to process chat";
      logEvent("ERROR /api/chat", { message, elapsedMs });
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/read-receipts", async (req, res) => {
    const startTime = Date.now();
    try {
      const { photos } = req.body as {
        photos: Array<{
          globalIndex: number;
          totalPhotos: number;
          mimeType: string;
          base64Data: string;
        }>;
      };

      if (!photos || !Array.isArray(photos) || photos.length === 0) {
        res.status(400).json({ error: "No photos provided in batch." });
        return;
      }

      if (!resolveApiKey()) {
        res.status(400).json({
          error:
            "Receipt photo scanning requires a Gemini API key. However, PDF chapter breakdown works 100% locally with Python (zero API key needed).",
        });
        return;
      }

      const indices = photos.map((p) => p.globalIndex);
      const totalKB = Math.round(
        photos.reduce((acc, p) => acc + (p.base64Data?.length || 0) * 0.75, 0) /
          1024
      );
      logEvent("START /api/read-receipts", {
        model: activeModelId,
        photoIndices: indices,
        totalKB,
      });

      const parts: Array<
        { text: string } | { inlineData: { mimeType: string; data: string } }
      > = [];

      const indexDescriptions = photos
        .map(
          (p, i) =>
            `Image ${i + 1} in this request is Photo ${p.globalIndex} of ${p.totalPhotos} (set photo_index = ${p.globalIndex}).`
        )
        .join("\n");

      parts.push({
        text: `Examine the following ${photos.length} photo(s) of receipts.\n${indexDescriptions}\n\nFind every physical receipt in each photo, box it with box_2d = [ymin, xmin, ymax, xmax] (0-1000), and extract all fields strictly following the schema.`,
      });

      for (const photo of photos) {
        parts.push({
          text: `Photo ${photo.globalIndex} of ${photo.totalPhotos} is the image below (use photo_index: ${photo.globalIndex}):`,
        });
        parts.push({
          inlineData: {
            mimeType: photo.mimeType || "image/jpeg",
            data: photo.base64Data,
          },
        });
      }

      const { response, modelUsed } = await generateWithModelFallback({
        contents: { parts },
        config: {
          systemInstruction: READ_SYSTEM_INSTRUCTION,
          responseMimeType: "application/json",
          responseSchema: READ_RESPONSE_SCHEMA,
          temperature: 0.1,
        },
      });

      const parsed = parseJsonSafely<{ receipts: unknown[] }>(response.text, {
        receipts: [],
      });
      const elapsedMs = Date.now() - startTime;
      logEvent("DONE /api/read-receipts", {
        model: modelUsed,
        photoIndices: indices,
        receiptsFound: Array.isArray(parsed.receipts)
          ? parsed.receipts.length
          : 0,
        elapsedMs,
      });

      res.json(parsed);
    } catch (error: unknown) {
      const elapsedMs = Date.now() - startTime;
      const message =
        error instanceof Error ? error.message : "Failed to read receipts batch";
      logEvent("ERROR /api/read-receipts", { message, elapsedMs });
      res.status(500).json({ error: message });
    }
  });

  app.post("/api/check-duplicates", async (req, res) => {
    const startTime = Date.now();
    try {
      const { rows } = req.body as {
        rows: Array<{
          row: number;
          photo_index: number;
          merchant: string | null;
          date: string | null;
          time: string | null;
          final_total: number | null;
          printed_total: number | null;
          currency: string | null;
          card_last4: string | null;
          source_type?: string;
          source_label?: string;
        }>;
      };

      if (!rows || !Array.isArray(rows) || rows.length < 2) {
        res.json({ duplicates: [] });
        return;
      }

      if (!resolveApiKey()) {
        logEvent("SKIP /api/check-duplicates: No GEMINI_API_KEY configured");
        res.json({ duplicates: [] });
        return;
      }

      logEvent("START /api/check-duplicates", {
        model: activeModelId,
        rowCount: rows.length,
      });

      const { response, modelUsed } = await generateWithModelFallback({
        contents: `Here are the merged receipt rows extracted from a pile of receipt photos and synced Workspace sources:\n\n${JSON.stringify(
          rows,
          null,
          2
        )}\n\nIdentify any likely duplicate pairs where the same transaction appears more than once (e.g., same merchant, same total, same or adjacent date, same card last 4 digits). Pay special attention to duplicate transactions across different ingestion sources (e.g., a receipt synced from Google Drive or Gmail that matches a manual upload or another email/Drive file) and include that cross-source context in "reason". Always set "row" to the higher row number and "duplicate_of" to the lower row number.`,
        config: {
          systemInstruction:
            "You are a careful bookkeeper checking extracted receipt rows for duplicate transactions across manual uploads, Google Drive folders, and Gmail receipts. Only report pairs that are genuinely likely to be the same transaction captured twice.",
          responseMimeType: "application/json",
          responseSchema: DUPLICATES_RESPONSE_SCHEMA,
          temperature: 0.1,
        },
      });

      const parsed = parseJsonSafely<{ duplicates: unknown[] }>(response.text, {
        duplicates: [],
      });
      logEvent("DONE /api/check-duplicates", {
        model: modelUsed,
        duplicatesFound: Array.isArray(parsed.duplicates)
          ? parsed.duplicates.length
          : 0,
        elapsedMs: Date.now() - startTime,
      });

      res.json(parsed);
    } catch (error: unknown) {
      const message =
        error instanceof Error ? error.message : "Failed to check duplicates";
      logEvent("ERROR /api/check-duplicates", {
        message,
        elapsedMs: Date.now() - startTime,
      });
      res.status(500).json({ error: message });
    }
  });

  // -----------------------------------------------------------------------
  // PDF Chapter Parsing Endpoint
  // -----------------------------------------------------------------------

  /** System instruction for chapter-by-chapter PDF parsing. */
  const PDF_PARSE_SYSTEM_INSTRUCTION = `You are an expert document structuring engine. Your task is to analyze an entire PDF document and extract clean, sequential chapter and section boundaries.

Guiding Principles:
1. Complete Document Coverage: Every page from page 1 to the final page must be accounted for across the chapters. There must be NO skipped pages and NO overlapping page ranges.
2. Structure Recognition:
   - If the document has a Table of Contents (TOC) or outline bookmarks, inspect ALL levels of the hierarchy. If chapters are nested under 'Parts' or 'Units', extract the individual chapters (or include standalone un-nested sections like Preface, Introduction, Appendix, Conclusion). Do NOT discard the TOC just because headings are across multiple indentation levels or have an overarching book title. Use explicit TOC entries as exact chapter boundaries. Set is_explicit_chapter = true.
   - Front-Matter & Back-Matter Recognition: Treat preliminary materials (Cover, Table of Contents, Preface, Foreword, Acknowledgments, Dedication, Executive Summary) as distinct front-matter sections starting on page 1 rather than lumping them into Chapter 1. Similarly, treat reference sections (Appendices, Bibliography, References, Works Cited, Index, Glossary, Endnotes) as distinct back-matter sections rather than swallowing them into the final chapter.
   - FALLBACK (Unstructured Documents): Only if the document genuinely has NO Table of Contents, headings, or structural divisions, semantically partition it into 3 to 10 balanced chapters based on topical progression. Set is_explicit_chapter = false.
3. For each chapter:
   - title: Clear, concise chapter title.
   - start_page: Starting page number in the PDF (1-based integer).
   - end_page: Ending page number in the PDF (1-based integer, inclusive).
   - summary: High-value 2-4 sentence summary of core insights.
   - content: Detailed chapter overview (100–300 words), highlighting key arguments and section anchors.
   - key_topics: 2-5 core concepts or tags.
   - word_count: Word count estimate for this chapter.
4. Strict Ordering: Chapters must be ordered sequentially (chapter_number = 1, 2, 3...).
5. If the PDF is completely blank or corrupt, return an empty chapters list with total_chapters = 0.
6. Scanned / Image-Only PDFs: If the document contains scanned images or photocopies without an embedded text layer, perform visual OCR on each page to read chapter titles, section boundaries, and page contents accurately.`;

  /** Structured response schema for PDF chapter parsing. */
  const PDF_PARSE_RESPONSE_SCHEMA = {
    type: Type.OBJECT,
    properties: {
      document_title: {
        type: Type.STRING,
        description:
          "The document title from the title page or inferred from content.",
      },
      total_pages: {
        type: Type.NUMBER,
        description: "Total number of pages in the PDF.",
      },
      total_chapters: {
        type: Type.NUMBER,
        description: "Total number of chapters or sections found/inferred.",
      },
      has_explicit_structure: {
        type: Type.BOOLEAN,
        description:
          "True if the document has explicit chapter/section headings; false if structure was inferred.",
      },
      chapters: {
        type: Type.ARRAY,
        items: {
          type: Type.OBJECT,
          properties: {
            chapter_number: {
              type: Type.NUMBER,
              description: "1-based chapter/section number in document order.",
            },
            title: {
              type: Type.STRING,
              description:
                "Title of the chapter/section as printed or inferred.",
            },
            start_page: {
              type: Type.NUMBER,
              description: "Starting page number (1-based).",
            },
            end_page: {
              type: Type.NUMBER,
              description: "Ending page number (1-based, inclusive).",
            },
            content: {
              type: Type.STRING,
              description:
                "Detailed chapter overview highlighting core concepts and anchors.",
            },
            summary: {
              type: Type.STRING,
              description: "Brief 2-4 sentence summary of this chapter.",
            },
            key_topics: {
              type: Type.ARRAY,
              items: { type: Type.STRING },
              description: "2-5 key topics or themes in this chapter.",
            },
            word_count: {
              type: Type.NUMBER,
              description: "Estimated number of words in this chapter.",
            },
            is_explicit_chapter: {
              type: Type.BOOLEAN,
              description:
                "True if chapter boundary came from explicit document structure; false if inferred.",
            },
          },
          required: [
            "chapter_number",
            "title",
            "start_page",
            "end_page",
            "content",
            "summary",
            "key_topics",
            "word_count",
            "is_explicit_chapter",
          ],
        },
      },
    },
    required: [
      "document_title",
      "total_pages",
      "total_chapters",
      "has_explicit_structure",
      "chapters",
    ],
  };

  interface NormalizedChapter {
    chapter_number: number;
    title: string;
    chapter_title: string;
    start_page: number;
    end_page: number;
    content: string;
    extracted_text: string;
    summary: string;
    key_topics: string[];
    word_count: number;
    is_explicit_chapter: boolean;
    section_type?: "front_matter" | "chapter" | "back_matter";
    is_scanned?: boolean;
  }

  function validateAndRepairChapters(
    rawChapters: unknown[],
    reportedTotalPages: number,
    fallbackTitle: string
  ): { chapters: NormalizedChapter[]; totalPages: number } {
    if (!Array.isArray(rawChapters) || rawChapters.length === 0) {
      const singleTotal = Math.max(
        1,
        Math.floor(Number(reportedTotalPages) || 1)
      );
      return {
        chapters: [
          {
            chapter_number: 1,
            title: fallbackTitle || "Complete Document",
            chapter_title: fallbackTitle || "Complete Document",
            start_page: 1,
            end_page: singleTotal,
            content: "Full document view.",
            extracted_text: "Full document view.",
            summary: "Complete document read.",
            key_topics: ["Overview"],
            word_count: 0,
            is_explicit_chapter: false,
          },
        ],
        totalPages: singleTotal,
      };
    }

    const sanitized: NormalizedChapter[] = rawChapters.map(
      (ch: any, idx: number) => {
        const title =
          typeof ch?.title === "string" && ch.title.trim()
            ? ch.title.trim()
            : typeof ch?.chapter_title === "string" && ch.chapter_title.trim()
            ? ch.chapter_title.trim()
            : `Chapter ${idx + 1}`;

        const rawStart = Number(ch?.start_page);
        const rawEnd = Number(ch?.end_page);
        const start_page =
          Number.isFinite(rawStart) && rawStart >= 1
            ? Math.floor(rawStart)
            : idx + 1;
        const end_page =
          Number.isFinite(rawEnd) && rawEnd >= start_page
            ? Math.floor(rawEnd)
            : start_page;

        const summary =
          typeof ch?.summary === "string" && ch.summary.trim()
            ? ch.summary.trim()
            : `Chapter ${idx + 1} content.`;

        const text =
          typeof ch?.extracted_text === "string" && ch.extracted_text.trim()
            ? ch.extracted_text.trim()
            : typeof ch?.content === "string" && ch.content.trim()
            ? ch.content.trim()
            : summary;

        const key_topics =
          Array.isArray(ch?.key_topics) && ch.key_topics.length > 0
            ? ch.key_topics.map((t: any) => String(t).trim()).filter(Boolean)
            : [title];

        const word_count =
          typeof ch?.word_count === "number" && !Number.isNaN(ch.word_count)
            ? Math.max(0, Math.floor(ch.word_count))
            : text.split(/\s+/).filter(Boolean).length;

        const is_explicit_chapter = Boolean(ch?.is_explicit_chapter ?? true);
        const section_type = (ch as any)?.section_type as 'front_matter' | 'chapter' | 'back_matter' | undefined;
        const is_scanned = Boolean((ch as any)?.is_scanned);

        const extractedNum =
          normalizeChapterNumber(ch?.chapter_number) ??
          normalizeChapterNumber(title) ??
          idx + 1;

        return {
          chapter_number: extractedNum,
          title,
          chapter_title: title,
          start_page,
          end_page,
          content: text,
          extracted_text: text,
          summary,
          key_topics,
          word_count,
          is_explicit_chapter,
          section_type,
          is_scanned,
        };
      }
    );

    // Sort chapters strictly in ascending numerical order
    const sorted = sortChapterItems(sanitized);

    // Front-Matter gap check: If first chapter doesn't start at page 1, create a Front Matter section rather than bloating Chapter 1
    if (sorted.length > 0 && sorted[0].start_page > 1) {
      const frontEnd = sorted[0].start_page - 1;
      const frontMatterChapter: NormalizedChapter = {
        chapter_number: 1,
        title: "Front Matter & Preface",
        chapter_title: "Front Matter & Preface",
        start_page: 1,
        end_page: frontEnd,
        content: `Preliminary material, table of contents, and front matter covering pages 1 to ${frontEnd}.`,
        extracted_text: "",
        summary: `Preliminary material, table of contents, and front matter covering pages 1 to ${frontEnd}.`,
        key_topics: ["Front Matter", "Preface", "Table of Contents"],
        word_count: 0,
        is_explicit_chapter: false,
        section_type: "front_matter",
        is_scanned: sorted.length > 0 ? sorted[0].is_scanned : false,
      };
      sorted.unshift(frontMatterChapter);
      for (let i = 0; i < sorted.length; i++) {
        if (!sorted[i].is_explicit_chapter || sorted[i].section_type === "front_matter") {
          sorted[i].chapter_number = i + 1;
        }
      }
    }

    // ---------------------------------------------------------------
    // Contiguous page-range repair: fix gaps & overlaps between adjacent chapters
    // ---------------------------------------------------------------
    // After this pass every page from 1..totalPages is covered exactly once.
    // Rules:
    //  • If ch[i].end_page < ch[i].start_page → clamp end to start (degenerate).
    //  • Overlap: ch[i].end_page >= ch[i+1].start_page → trim ch[i].end_page
    //    to ch[i+1].start_page - 1 (the later chapter's boundary wins),
    //    or keep ch[i].start_page if multiple sub-sections share the same page.
    //  • Gap: ch[i].end_page + 1 < ch[i+1].start_page → extend ch[i].end_page
    //    to ch[i+1].start_page - 1 so no pages are orphaned.
    for (let i = 0; i < sorted.length; i++) {
      // Fix degenerate end < start within the same chapter
      if (sorted[i].end_page < sorted[i].start_page) {
        sorted[i].end_page = sorted[i].start_page;
      }

      if (i + 1 < sorted.length) {
        const nextStart = sorted[i + 1].start_page;

        // Overlap: current chapter's end bleeds into the next chapter
        if (sorted[i].end_page >= nextStart) {
          sorted[i].end_page = nextStart > sorted[i].start_page ? nextStart - 1 : sorted[i].start_page;
        }

        // Gap: pages between current end and next start are unaccounted for
        if (sorted[i].end_page + 1 < nextStart) {
          sorted[i].end_page = nextStart - 1;
        }
      }
    }

    // Ensure the last chapter extends to the reported total pages
    let maxEnd = Math.max(reportedTotalPages, 1);
    for (const ch of sorted) {
      maxEnd = Math.max(maxEnd, ch.end_page);
    }

    // Back-matter gap check: If significant unrepresented pages exist after the final chapter, create distinct Back Matter
    if (sorted.length > 0 && reportedTotalPages > 0) {
      const lastChapter = sorted[sorted.length - 1];
      if (lastChapter.end_page < reportedTotalPages) {
        const gap = reportedTotalPages - lastChapter.end_page;
        if (gap >= 4) {
          const backStart = lastChapter.end_page + 1;
          sorted.push({
            chapter_number: sorted.length + 1,
            title: "Back Matter & Index",
            chapter_title: "Back Matter & Index",
            start_page: backStart,
            end_page: reportedTotalPages,
            content: `Reference materials, appendices, notes, or index covering pages ${backStart} to ${reportedTotalPages}.`,
            extracted_text: "",
            summary: `Reference materials, appendices, notes, or index covering pages ${backStart} to ${reportedTotalPages}.`,
            key_topics: ["Back Matter", "Index", "References"],
            word_count: 0,
            is_explicit_chapter: false,
            section_type: "back_matter",
            is_scanned: lastChapter.is_scanned,
          });
        } else {
          lastChapter.end_page = reportedTotalPages;
        }
      }
    }

    // Sequential chapter_number pass for unstructured documents without explicit author numbers
    const hasExplicitNumbers = sorted.some(
      (c) => c.is_explicit_chapter && typeof c.chapter_number === "number" && c.chapter_number > 0 && c.section_type !== "front_matter"
    );
    if (!hasExplicitNumbers) {
      for (let i = 0; i < sorted.length; i++) {
        sorted[i].chapter_number = i + 1;
      }
    }

    const calculatedTotal = Math.max(reportedTotalPages, maxEnd, sorted.length > 0 ? sorted[sorted.length - 1].end_page : 1);
    return { chapters: sorted, totalPages: calculatedTotal };
  }

  /**
   * Pure JavaScript PDF structure extractor using pdf-lib.
   * Runs inside Node.js with zero external binary dependencies and zero API keys.
   * Used when Python/fitz/pypdf are unavailable and no Gemini API key is configured.
   */
  async function parsePdfWithPdfLib(
    pdfBuffer: Buffer,
    fileName?: string
  ): Promise<{
    document_title: string;
    total_pages: number;
    total_chapters: number;
    has_explicit_structure: boolean;
    chapters: NormalizedChapter[];
  }> {
    const doc = await PDFDocument.load(pdfBuffer, { ignoreEncryption: true });
    const totalPages = Math.max(1, doc.getPageCount());

    let metaTitle = doc.getTitle()?.trim() || "";
    if (metaTitle.toLowerCase().endsWith(".pdf")) {
      metaTitle = metaTitle.slice(0, -4);
    }
    const cleanDocTitle =
      metaTitle ||
      (fileName ? fileName.replace(/\.[a-zA-Z0-9]+$/, "").replace(/[-_]/g, " ").trim() : "") ||
      "Untitled Document";

    // Partition document into balanced logical chapters covering 100% of pages
    const numChapters =
      totalPages <= 1
        ? 1
        : totalPages <= 4
        ? totalPages
        : Math.min(10, Math.max(3, Math.round(totalPages / 6)));

    const rawChapters: NormalizedChapter[] = [];
    const pagesPerChapter = Math.max(1, Math.floor(totalPages / numChapters));

    for (let i = 0; i < numChapters; i++) {
      const sp = i === 0 ? 1 : i * pagesPerChapter + 1;
      const ep = i === numChapters - 1 ? totalPages : Math.min(totalPages, (i + 1) * pagesPerChapter);
      const chNum = i + 1;

      let partTitle = `Section ${chNum}: Overview & Analysis (Pages ${sp}–${ep})`;
      let secType: "front_matter" | "chapter" | "back_matter" = "chapter";

      if (numChapters > 2) {
        if (i === 0 && sp === 1) {
          partTitle = `Part 1: Preliminary & Overview (Pages ${sp}–${ep})`;
          secType = "front_matter";
        } else if (i === numChapters - 1 && ep === totalPages) {
          partTitle = `Part ${chNum}: Concluding Synthesis (Pages ${sp}–${ep})`;
          secType = "back_matter";
        } else {
          partTitle = `Part ${chNum}: Core Analysis (Pages ${sp}–${ep})`;
          secType = "chapter";
        }
      } else if (numChapters === 1) {
        partTitle = `${cleanDocTitle} (Complete Document, Pages 1–${totalPages})`;
      }

      rawChapters.push({
        chapter_number: chNum,
        title: partTitle,
        chapter_title: partTitle,
        start_page: sp,
        end_page: ep,
        content: `Extracted logical section ${chNum} of ${cleanDocTitle}, covering pages ${sp} to ${ep}.`,
        extracted_text: `Extracted logical section ${chNum} of ${cleanDocTitle}, covering pages ${sp} to ${ep}.`,
        summary: `Logical section ${chNum} covering document progression from page ${sp} through page ${ep}.`,
        key_topics: [cleanDocTitle, `Pages ${sp}–${ep}`],
        word_count: 0,
        is_explicit_chapter: false,
        section_type: secType,
      });
    }

    const { chapters: normalizedChapters, totalPages: calculatedTotal } =
      validateAndRepairChapters(rawChapters, totalPages, cleanDocTitle);

    return {
      document_title: cleanDocTitle,
      total_pages: calculatedTotal,
      total_chapters: normalizedChapters.length,
      has_explicit_structure: false,
      chapters: normalizedChapters,
    };
  }

  /**
   * POST /api/parse-pdf-chapters
   *
   * Accepts a base64-encoded PDF and returns structured chapter-by-chapter
   * data parsed locally via Python (PyMuPDF / pypdf) with zero API keys required,
   * with fallback to Gemini if an API key is configured.
   *
   * Request body: { mimeType: string, base64Data: string, fileName?: string }
   * Response:     ParsedPdfResult JSON
   */
  app.post("/api/parse-pdf-chapters", async (req, res) => {
    const startTime = Date.now();
    try {
      const { mimeType, base64Data, fileName } = req.body as {
        mimeType: string;
        base64Data: string;
        fileName?: string;
      };

      // --- Validate input ---
      if (!base64Data || typeof base64Data !== "string") {
        res.status(400).json({ error: "No PDF data provided." });
        return;
      }

      // Clean base64 data: strip any data URL prefix and whitespace
      const cleanBase64 = base64Data.includes(",")
        ? base64Data.slice(base64Data.indexOf(",") + 1).trim()
        : base64Data.trim();

      if (!cleanBase64) {
        res.status(400).json({ error: "No PDF data provided." });
        return;
      }

      const effectiveMime = (mimeType || "").toLowerCase().trim();
      const isPdfLike =
        !effectiveMime ||
        effectiveMime.includes("pdf") ||
        effectiveMime === "application/octet-stream" ||
        (Boolean(fileName) && fileName!.toLowerCase().endsWith(".pdf"));

      if (!isPdfLike) {
        res.status(400).json({
          error: `Invalid MIME type "${mimeType}". Expected a PDF document.`,
        });
        return;
      }

      let pdfBuffer: Buffer;
      try {
        pdfBuffer = Buffer.from(cleanBase64, "base64");
      } catch {
        res.status(400).json({ error: "Failed to decode base64 PDF payload." });
        return;
      }

      // Verify PDF magic header in first 1024 bytes per ISO 32000-1 §7.5.2
      const headerChunk = pdfBuffer.slice(0, 1024).toString("latin1");
      if (!headerChunk.includes("%PDF-")) {
        res.status(400).json({
          error: "The provided file is not a valid PDF document (missing %PDF header).",
        });
        return;
      }

      const pdfSizeKB = Math.round(pdfBuffer.length / 1024);
      logEvent("START /api/parse-pdf-chapters", {
        method: "hybrid-local",
        fileName: fileName || "unknown.pdf",
        pdfSizeKB,
      });

      // Quick metadata probe using pdf-lib (runs in <10ms)
      let accuratePageCount = 1;
      let accurateTitle = fileName || "Document";
      try {
        const preDoc = await PDFDocument.load(pdfBuffer, { ignoreEncryption: true });
        accuratePageCount = Math.max(1, preDoc.getPageCount());
        const t = preDoc.getTitle()?.trim();
        if (t) {
          accurateTitle = t.toLowerCase().endsWith(".pdf") ? t.slice(0, -4) : t;
        }
      } catch (preErr) {
        logEvent("WARN pdf-lib pre-check notice", { err: String(preErr) });
      }

      // --- Execute Local Python Extraction (0 API keys required) ---
      const tempId = `shoebox_${Date.now()}_${Math.random().toString(36).substring(2, 9)}.pdf`;
      const tempFilePath = path.join(os.tmpdir(), tempId);
      const scriptPath = path.join(__dirname, "scripts", "pdf_chapter_extractor.py");

      let pyResult: any = null;
      let pythonSuccess = false;

      try {
        await fs.promises.writeFile(tempFilePath, pdfBuffer);

        const { stdout } = await execFileAsync(
          "python3",
          [scriptPath, tempFilePath, accurateTitle || fileName || "Document"],
          { maxBuffer: 64 * 1024 * 1024 }
        );

        if (stdout && stdout.trim()) {
          pyResult = JSON.parse(stdout.trim());
          if (pyResult && Array.isArray(pyResult.chapters) && pyResult.chapters.length > 0) {
            pythonSuccess = true;
          }
        }
      } catch (pyErr: any) {
        logEvent("WARN python-extractor failed, evaluating fallback", {
          err: String(pyErr?.message || pyErr),
        });
      } finally {
        try {
          if (fs.existsSync(tempFilePath)) {
            await fs.promises.unlink(tempFilePath);
          }
        } catch {
          // ignore cleanup error
        }
      }

      const totalPages = Math.max(accuratePageCount, pyResult?.total_pages || 1);
      const isScannedPdf = Boolean(
        pyResult?.is_scanned_pdf ||
          (pyResult?.total_words !== undefined && pyResult.total_words === 0)
      );

      // If Python successfully parsed an electronic document with embedded text, return immediately:
      if (pythonSuccess && pyResult && !isScannedPdf) {
        const { chapters: normalizedChapters, totalPages: calculatedTotal } =
          validateAndRepairChapters(
            pyResult.chapters || [],
            totalPages,
            accurateTitle || fileName || "Document"
          );

        const result = {
          document_title:
            pyResult.document_title || accurateTitle || fileName || "Untitled Document",
          total_pages: calculatedTotal,
          total_chapters: normalizedChapters.length,
          has_explicit_structure: Boolean(pyResult.has_explicit_structure ?? true),
          is_scanned_pdf: false,
          ocr_recommended: false,
          total_words: pyResult.total_words,
          chapters: normalizedChapters,
        };

        const elapsedMs = Date.now() - startTime;
        logEvent("DONE /api/parse-pdf-chapters (Python Local Text)", {
          method: "python-local",
          fileName: fileName || "unknown.pdf",
          chaptersFound: normalizedChapters.length,
          totalPages: result.total_pages,
          elapsedMs,
        });

        res.json(result);
        return;
      }

      // --- Optional Fallback to Gemini IF API key is present ---
      if (resolveApiKey()) {
        try {
          logEvent(
            isScannedPdf
              ? "ROUTE /api/parse-pdf-chapters to Gemini Vision OCR (scanned PDF detected)"
              : "FALLBACK /api/parse-pdf-chapters to Gemini",
            {
              fileName: fileName || "unknown.pdf",
              isScannedPdf,
            }
          );

          const ocrHint = isScannedPdf
            ? " This PDF appears to be scanned or image-only with no embedded text layer. Use visual OCR to read every page, then extract chapters from the recognized text."
            : "";

          const parts: Array<
            { text: string } | { inlineData: { mimeType: string; data: string } }
          > = [
            {
              text: `Read the following PDF document in its entirety. Parse and return every chapter or logical section in strict document order, following the schema exactly.${
                fileName ? ` The file is named "${fileName}".` : ""
              }${ocrHint}`,
            },
            {
              inlineData: {
                mimeType: "application/pdf",
                data: cleanBase64,
              },
            },
          ];

          const { response, modelUsed } = await generateWithModelFallback({
            contents: { parts },
            config: {
              systemInstruction: PDF_PARSE_SYSTEM_INSTRUCTION,
              responseMimeType: "application/json",
              responseSchema: PDF_PARSE_RESPONSE_SCHEMA,
              temperature: 0.1,
            },
          });

          const parsed = parseJsonSafely<{
            document_title?: string;
            total_pages?: number;
            total_chapters?: number;
            has_explicit_structure?: boolean;
            chapters?: unknown[];
          }>(response.text, {
            document_title: fileName || "Untitled Document",
            total_pages: 0,
            total_chapters: 0,
            has_explicit_structure: false,
            chapters: [],
          });

          const { chapters: normalizedChapters, totalPages: calculatedTotal } =
            validateAndRepairChapters(
              parsed.chapters || [],
              Math.max(totalPages, parsed.total_pages || 0),
              accurateTitle || fileName || "Document"
            );

          const result = {
            document_title:
              parsed.document_title || accurateTitle || fileName || "Untitled Document",
            total_pages: calculatedTotal,
            total_chapters: normalizedChapters.length,
            has_explicit_structure: Boolean(parsed.has_explicit_structure ?? true),
            chapters: normalizedChapters,
          };

          const elapsedMs = Date.now() - startTime;
          logEvent("DONE /api/parse-pdf-chapters (Gemini Fallback)", {
            model: modelUsed,
            fileName: fileName || "unknown.pdf",
            chaptersFound: normalizedChapters.length,
            totalPages: result.total_pages,
            elapsedMs,
          });

          res.json(result);
          return;
        } catch (geminiErr) {
          logEvent("WARN Gemini fallback failed or quota exhausted, using local engine", {
            err: String(geminiErr),
          });
        }
      }

      // --- If Python partitioned chapters successfully, use them directly ---
      if (pythonSuccess && pyResult && Array.isArray(pyResult.chapters) && pyResult.chapters.length > 0) {
        const { chapters: normalizedChapters, totalPages: calculatedTotal } =
          validateAndRepairChapters(
            pyResult.chapters,
            totalPages,
            accurateTitle || fileName || "Document"
          );

        const result = {
          document_title:
            pyResult.document_title || accurateTitle || fileName || "Untitled Document",
          total_pages: calculatedTotal,
          total_chapters: normalizedChapters.length,
          has_explicit_structure: Boolean(pyResult.has_explicit_structure ?? true),
          is_scanned_pdf: Boolean(pyResult.is_scanned_pdf),
          ocr_recommended: Boolean(pyResult.ocr_recommended),
          total_words: pyResult.total_words,
          chapters: normalizedChapters,
        };

        const elapsedMs = Date.now() - startTime;
        logEvent("DONE /api/parse-pdf-chapters (Python Scanned/Partitioned Recovery)", {
          method: "python-local",
          fileName: fileName || "unknown.pdf",
          chaptersFound: normalizedChapters.length,
          totalPages: result.total_pages,
          elapsedMs,
        });

        res.json(result);
        return;
      }

      // --- Zero-Dependency Pure JS Fallback via pdf-lib (0 API keys, 0 Python required) ---
      try {
        logEvent("FALLBACK /api/parse-pdf-chapters to pure JS pdf-lib engine", {
          fileName: fileName || "unknown.pdf",
        });
        const jsResult = await parsePdfWithPdfLib(pdfBuffer, accurateTitle || fileName);

        const elapsedMs = Date.now() - startTime;
        logEvent("DONE /api/parse-pdf-chapters (pdf-lib Pure JS Fallback)", {
          fileName: fileName || "unknown.pdf",
          chaptersFound: jsResult.total_chapters,
          totalPages: jsResult.total_pages,
          elapsedMs,
        });

        res.json(jsResult);
        return;
      } catch (pdfLibErr) {
        logEvent("WARN pdf-lib pure JS fallback failed", {
          err: String(pdfLibErr),
        });
      }

      // If all else fails, synthesize structured chapters using pyResult or minimal bounds
      const fallbackTotalPages = totalPages || 1;
      const { chapters: fallbackChapters, totalPages: calculatedFallbackTotal } =
        validateAndRepairChapters(
          pyResult?.chapters || [],
          fallbackTotalPages,
          accurateTitle || fileName || "Document"
        );

      const result = {
        document_title:
          pyResult?.document_title || accurateTitle || fileName || "Untitled Document",
        total_pages: calculatedFallbackTotal,
        total_chapters: fallbackChapters.length,
        has_explicit_structure: Boolean(pyResult?.has_explicit_structure ?? false),
        is_scanned_pdf: Boolean(pyResult?.is_scanned_pdf),
        ocr_recommended: Boolean(pyResult?.ocr_recommended),
        total_words: pyResult?.total_words,
        chapters: fallbackChapters,
      };

      const elapsedMs = Date.now() - startTime;
      logEvent("DONE /api/parse-pdf-chapters (Synthesized Fallback)", {
        fileName: fileName || "unknown.pdf",
        chaptersFound: fallbackChapters.length,
        totalPages: result.total_pages,
        elapsedMs,
      });

      res.json(result);
      return;
    } catch (error: unknown) {
      const elapsedMs = Date.now() - startTime;
      const message = extractHumanErrorMessage(
        error,
        "Failed to parse PDF document. Please verify the file is a valid PDF."
      );
      logEvent("ERROR /api/parse-pdf-chapters", { message, elapsedMs });
      res.status(500).json({ error: message });
    }
  });

  if (process.env.NODE_ENV !== "production") {

    const vite = await createViteServer({
      server: { middlewareMode: true },
      appType: "spa",
    });
    app.use(vite.middlewares);
  } else {
    const distPath = path.join(__dirname, "dist");
    app.use(express.static(distPath));
    app.get("*", (_req, res) => {
      res.sendFile(path.join(distPath, "index.html"));
    });
  }

  app.listen(PORT, "0.0.0.0", () => {
    logEvent(`Shoebox server running on http://localhost:${PORT}`);
  });
}

startServer();

require('dotenv').config();
const express = require('express');
const path = require('path');
const axios = require('axios');
const { GoogleGenerativeAI } = require('@google/generative-ai');
const OpenAI = require('openai');

const app = express();
const port = 3000;

// --- MIDDLEWARE ---
app.use(express.json());
app.use(express.static(path.join(__dirname)));

// --- API CLIENT INITIALIZATION ---
const genAI = new GoogleGenerativeAI(process.env.GEMINI_API_KEY);
const geminiModel = genAI.getGenerativeModel({ model: "gemini-2.5-flash" });
const groq = new OpenAI({ baseURL: "https://api.groq.com/openai/v1", apiKey: process.env.GROQ_API_KEY });

// NEW: OpenRouter Client
const openRouter = new OpenAI({
  // FIXED: Base URL should NOT include /chat/completions
  baseURL: "https://openrouter.ai/api/v1",
  apiKey: process.env.OPENROUTER_API_KEY,
  defaultHeaders: {
    "HTTP-Referer": "http://localhost:3000", // Optional, for OpenRouter rankings
    "X-Title": "Multi-Modal AI App", // Optional
  }
});

// --- SCORING FUNCTIONS ---
function calculateLexicalDensity(text) {
    const words = text.match(/\b(\w+)\b/g) || [];
    if (words.length === 0) return 0;
    const contentWordRegex = /\b(?!is|are|was|were|a|an|the|and|or|but|to|of|in|for|on|with|at|by|from|up|about|into|over|after)\w+\b/gi;
    const contentWords = text.match(contentWordRegex) || [];
    return (contentWords.length / words.length) * 100;
}

function calculateScore(text) {
    const words = text.match(/\b(\w+)\b/g) || [];
    const sentences = text.match(/[^.!?]+[.!?]+/g) || [];
    const wordCount = words.length;
    const sentenceCount = sentences.length;
    if (wordCount === 0 || sentenceCount === 0) return 0;
    const avgSentenceLength = wordCount / sentenceCount;
    const densityScore = calculateLexicalDensity(text);
    const finalScore = (50 + densityScore) - avgSentenceLength;
    return Math.round(finalScore);
}

// --- HELPER: CREATE IMAGE PROMPT ---
async function createImagePrompt(userQuery) {
    console.log("Generating image prompt...");
    try {
        const instruction = `Create a vivid, highly descriptive image prompt based on this text: "${userQuery}". Output ONLY the prompt, no other text.`;
        const completion = await groq.chat.completions.create({
            model: "llama-3.1-8b-instant",
            messages: [{ "role": "user", "content": instruction }],
        });
        return completion.choices[0].message.content.replace(/"/g, '');
    } catch (error) {
        console.error("Prompt generation failed:", error.message);
        return userQuery;
    }
}

// --- MAIN ENDPOINT ---
app.post('/generate-all', async (req, res) => {
    try {
        const { query: userQuery, models } = req.body;
        if (!userQuery) return res.status(400).json({ error: 'Query is required' });

        // 1. Gemini Task
        const askGemini = async () => {
            try {
                const result = await geminiModel.generateContent(userQuery);
                const response = await result.response;
                return { model: 'Gemini 2.5 Flash', answer: response.text() };
            } catch (error) {
                console.error("Gemini Error:", error.message);
                return { model: 'Gemini 2.5 Flash', answer: `Error: ${error.message}`, error: true };
            }
        };

        // 2. Groq Task
        const askGroq = async () => {
            try {
                const completion = await groq.chat.completions.create({
                    model: "llama-3.1-8b-instant",
                    messages: [{ "role": "user", "content": userQuery }]
                });
                return { model: 'Groq (Llama 3.1)', answer: completion.choices[0].message.content };
            } catch (error) {
                console.error("Groq Error:", error.message);
                return { model: 'Groq (Llama 3.1)', answer: `Error: ${error.message}`, error: true };
            }
        };

        // 3. OpenRouter Task [NEW]
        const askOpenRouter = async () => {
            try {
                const completion = await openRouter.chat.completions.create({
                    // Switched to Google's reliable free model on OpenRouter
                    model: "meta-llama/llama-3.2-3b-instruct:free", 
                    messages: [{ "role": "user", "content": userQuery }],
                });
                return { model: 'Llama 3.2 (OpenRouter)', answer: completion.choices[0].message.content };
            } catch (error) {
                console.error("OpenRouter Error:", error.message);
                return { model: 'Llama 3.2 (OpenRouter)', answer: `Error: ${error.message}`, error: true };
            }
        };

        // 4. Image Task (Pollinations)
        const generateImage = async () => {
            try {
                const imagePrompt = await createImagePrompt(userQuery);
                const encodedPrompt = encodeURIComponent(imagePrompt);
                const imageUrl = `https://image.pollinations.ai/prompt/${encodedPrompt}?width=1024&height=1024&nologo=true`;
                const response = await axios.get(imageUrl, { responseType: 'arraybuffer' });
                const base64Image = Buffer.from(response.data, 'binary').toString('base64');
                return { type: 'image', data: base64Image };
            } catch (error) {
                console.error("Image Gen Error:", error.message);
                return { type: 'image', data: null, error: true, message: "Failed to generate image." };
            }
        };

        // --- Build Task List ---
        const tasks = [];
        if (models.gemini) tasks.push(askGemini());
        if (models.groq) tasks.push(askGroq());
        if (models.openrouter) tasks.push(askOpenRouter()); // Updated key name
        if (models.image) tasks.push(generateImage());

        if (tasks.length === 0) return res.json({ textResults: [], imageResult: null });

        // --- Execute & Process ---
        const results = await Promise.all(tasks);
        const textResults = [];
        let imageResult = null;

        results.forEach(result => {
            if (result.type === 'image') {
                if (!result.error) imageResult = result.data;
            } else {
                const score = result.error ? -999 : calculateScore(result.answer);
                textResults.push({ ...result, score });
            }
        });

        res.json({ textResults, imageResult });

    } catch (error) {
        console.error("Server Error:", error);
        res.status(500).json({ error: 'Internal Server Error' });
    }
});

app.listen(port, () => {
    console.log(`🚀 Server running at http://localhost:${port}`);
});
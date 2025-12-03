import { GoogleGenAI, GenerateContentResponse, Type } from "@google/genai";
import { Job } from "../types";

// Initialize the Gemini API client
// The API key is securely retrieved from the environment variable.
const ai = new GoogleGenAI({ apiKey: process.env.API_KEY });

const MODEL_NAME = "gemini-2.5-flash";

export interface LogAnalysis {
  summary: string;
  rootCause: string;
  recommendation: string;
}

export const analyzeJobFailure = async (job: Job): Promise<LogAnalysis> => {
  if (!process.env.API_KEY) {
    throw new Error("API Key is missing. Please check your environment configuration.");
  }

  // Construct a prompt that includes the job context and the error logs
  const prompt = `
    You are a DevOps and Machine Learning Infrastructure expert.
    Analyze the following job failure.
    
    Job ID: ${job.id}
    Job Type: ${job.type}
    Hydra Config Overrides: ${job.hydra.overrides.join(" ")}
    
    STDERR LOG:
    \`\`\`
    ${job.logs.stderr.slice(-2000)}
    \`\`\`

    STDOUT TAIL:
    \`\`\`
    ${job.logs.stdout.slice(-500)}
    \`\`\`

    Provide the output in JSON format with the following structure:
    {
      "summary": "A brief 1-sentence summary of what happened.",
      "rootCause": "Technical explanation of the error (e.g., OOM, Syntax Error, Slurm Preemption).",
      "recommendation": "Actionable advice to fix the issue."
    }
  `;

  try {
    const response: GenerateContentResponse = await ai.models.generateContent({
      model: MODEL_NAME,
      contents: prompt,
      config: {
        responseMimeType: "application/json",
        responseSchema: {
          type: Type.OBJECT,
          properties: {
            summary: { type: Type.STRING },
            rootCause: { type: Type.STRING },
            recommendation: { type: Type.STRING }
          }
        }
      }
    });

    const text = response.text;
    if (!text) throw new Error("Empty response from Gemini");
    
    return JSON.parse(text) as LogAnalysis;

  } catch (error) {
    console.error("Gemini analysis failed:", error);
    throw new Error("Failed to analyze logs. Please try again.");
  }
};

export const askChatbot = async (history: {role: 'user'|'model', parts: [{text: string}]}[], newMessage: string) => {
   if (!process.env.API_KEY) {
    throw new Error("API Key missing");
  }

  const chat = ai.chats.create({
    model: "gemini-2.5-flash",
    history: history,
    config: {
      systemInstruction: "You are a helpful assistant for a developer using Hydra and Slurm. Keep answers concise and technical."
    }
  });

  const result = await chat.sendMessage({ message: newMessage });
  return result.text;
};
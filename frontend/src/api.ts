import { ConversationRequest, ConversationResponse } from './types';

const API_BASE = '/api/v1';

export async function sendMessage(request: ConversationRequest): Promise<ConversationResponse> {
  const response = await fetch(`${API_BASE}/conversation`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify(request),
  });

  if (!response.ok) {
    throw new Error(`API error: ${response.status}`);
  }

  return response.json();
}

export async function resetConversation(sessionId: string): Promise<void> {
  await fetch(`${API_BASE}/conversation/reset`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
    },
    body: JSON.stringify({ session_id: sessionId }),
  });
}

export async function checkHealth(): Promise<{ status: string }> {
  const response = await fetch(`${API_BASE}/conversation/health`);
  return response.json();
}

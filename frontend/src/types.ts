export interface PatientInfo {
  age?: number;
  sex?: 'male' | 'female';
}

export interface ConversationRequest {
  session_id?: string;
  message: string;
  patient_info?: PatientInfo;
  image_base64?: string;
}

export interface ConversationResponse {
  session_id: string;
  response: string;
  current_agent: string;
  symptoms_collected: string[];
  risk_level: 'unknown' | 'routine' | 'elevated' | 'urgent' | 'emergency';
  triage_complete: boolean;
  turn_count: number;
  specialty_hint: string | null;
  suggested_actions: string[];
  warnings: string[];
}

export interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  agent?: string;
  riskLevel?: string;
  isEmergency?: boolean;
}

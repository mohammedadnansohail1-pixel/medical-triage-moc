import { useState, useRef, useEffect } from 'react';

interface PatientInfo {
  age?: number;
  sex?: 'male' | 'female';
}

interface ConversationResponse {
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

interface Message {
  id: string;
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  agent?: string;
  riskLevel?: string;
  isEmergency?: boolean;
}

const API_BASE = '/api/v1';

async function sendMessage(request: {
  session_id?: string;
  message: string;
  patient_info?: PatientInfo;
  image_base64?: string;
}): Promise<ConversationResponse> {
  const response = await fetch(`${API_BASE}/conversation`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(request),
  });
  if (!response.ok) throw new Error(`API error: ${response.status}`);
  return response.json();
}

export function Chat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [sessionId, setSessionId] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [patientInfo, setPatientInfo] = useState<PatientInfo>({ age: undefined, sex: undefined });
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [riskLevel, setRiskLevel] = useState<string>('unknown');
  const [currentAgent, setCurrentAgent] = useState<string>('');
  const [symptoms, setSymptoms] = useState<string[]>([]);
  const [triageComplete, setTriageComplete] = useState(false);
  
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() && !selectedImage) return;
    if (isLoading) return;

    const userMessage: Message = {
      id: Date.now().toString(),
      role: 'user',
      content: input || '📷 Image uploaded',
      timestamp: new Date(),
    };

    setMessages(prev => [...prev, userMessage]);
    setInput('');
    setIsLoading(true);

    try {
      const response = await sendMessage({
        session_id: sessionId || undefined,
        message: input,
        patient_info: patientInfo,
        image_base64: selectedImage || undefined,
      });

      setSessionId(response.session_id);
      setRiskLevel(response.risk_level);
      setCurrentAgent(response.current_agent);
      setSymptoms(response.symptoms_collected);
      setTriageComplete(response.triage_complete);

      const assistantMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: response.response,
        timestamp: new Date(),
        agent: response.current_agent,
        riskLevel: response.risk_level,
        isEmergency: response.risk_level === 'emergency',
      };

      setMessages(prev => [...prev, assistantMessage]);
      setSelectedImage(null);
    } catch (error) {
      console.error('Error sending message:', error);
      const errorMessage: Message = {
        id: (Date.now() + 1).toString(),
        role: 'assistant',
        content: 'Sorry, there was an error. Please check if the backend is running on port 8000.',
        timestamp: new Date(),
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleReset = () => {
    setMessages([]);
    setSessionId(null);
    setRiskLevel('unknown');
    setCurrentAgent('');
    setSymptoms([]);
    setTriageComplete(false);
    setSelectedImage(null);
  };

  const handleImageSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onloadend = () => {
        const base64 = (reader.result as string).split(',')[1];
        setSelectedImage(base64);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const riskColors: Record<string, string> = {
    unknown: 'bg-gray-100 text-gray-700',
    routine: 'bg-green-100 text-green-700',
    elevated: 'bg-yellow-100 text-yellow-700',
    urgent: 'bg-orange-100 text-orange-700',
    emergency: 'bg-red-100 text-red-700 animate-pulse',
  };

  const agentColors: Record<string, string> = {
    emergency: 'bg-red-100 text-red-800',
    cardiology: 'bg-pink-100 text-pink-800',
    dermatology: 'bg-purple-100 text-purple-800',
    supervisor: 'bg-blue-100 text-blue-800',
    triage: 'bg-green-100 text-green-800',
  };

  return (
    <div className="flex flex-col h-screen max-w-4xl mx-auto">
      {/* Header */}
      <header className="bg-white border-b px-4 py-3 flex items-center justify-between shadow-sm">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 bg-blue-600 rounded-xl flex items-center justify-center text-white text-xl">
            🏥
          </div>
          <div>
            <h1 className="font-semibold text-gray-900">Medical Triage AI</h1>
            <p className="text-xs text-gray-500">Multiagent Symptom Assessment</p>
          </div>
        </div>
        
        <div className="flex items-center gap-3">
          {riskLevel !== 'unknown' && (
            <div className={`px-3 py-1 rounded-full text-sm font-medium ${riskColors[riskLevel]}`}>
              ⚠️ {riskLevel.charAt(0).toUpperCase() + riskLevel.slice(1)}
            </div>
          )}
          <button
            onClick={handleReset}
            className="px-3 py-1.5 text-sm text-gray-600 hover:text-gray-800 hover:bg-gray-100 rounded-lg transition"
          >
            🔄 Reset
          </button>
        </div>
      </header>

      {/* Emergency Banner */}
      {riskLevel === 'emergency' && (
        <div className="bg-red-600 text-white px-4 py-3 flex items-center gap-2 animate-pulse">
          <span className="text-xl">🚨</span>
          <span className="font-semibold">EMERGENCY DETECTED - Please call 911 immediately</span>
        </div>
      )}

      {/* Patient Info */}
      {messages.length === 0 && (
        <div className="bg-blue-50 border-b border-blue-100 px-4 py-3">
          <div className="flex items-center gap-4 flex-wrap">
            <span className="text-sm font-medium text-blue-800">Patient Info (optional):</span>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Age:</label>
              <input
                type="number"
                min="0"
                max="120"
                value={patientInfo.age || ''}
                onChange={(e) => setPatientInfo({ ...patientInfo, age: parseInt(e.target.value) || undefined })}
                className="w-16 px-2 py-1 border rounded text-sm"
                placeholder="--"
              />
            </div>
            <div className="flex items-center gap-2">
              <label className="text-sm text-gray-600">Sex:</label>
              <select
                value={patientInfo.sex || ''}
                onChange={(e) => setPatientInfo({ ...patientInfo, sex: e.target.value as 'male' | 'female' | undefined })}
                className="px-2 py-1 border rounded text-sm"
              >
                <option value="">--</option>
                <option value="male">Male</option>
                <option value="female">Female</option>
              </select>
            </div>
          </div>
        </div>
      )}

      {/* Agent Status Bar */}
      {currentAgent && (
        <div className="bg-gray-50 border-b px-4 py-2 flex items-center gap-4 text-sm">
          <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${agentColors[currentAgent] || 'bg-gray-100'}`}>
            🤖 {currentAgent}
          </span>
          {symptoms.length > 0 && (
            <span className="text-gray-500">
              Symptoms: <span className="font-medium">{symptoms.join(', ')}</span>
            </span>
          )}
          {triageComplete && (
            <span className="bg-green-100 text-green-700 px-2 py-0.5 rounded text-xs font-medium">
              ✅ Triage Complete
            </span>
          )}
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto p-4 space-y-4 bg-gray-50">
        {messages.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full text-center text-gray-500">
            <div className="text-5xl mb-4">👋</div>
            <h2 className="text-lg font-medium text-gray-700 mb-2">Welcome to Medical Triage AI</h2>
            <p className="max-w-md text-sm mb-6">
              Describe your symptoms and I will help assess your condition. 
              You can also upload images for skin-related concerns.
            </p>
            <div className="flex flex-wrap gap-2 justify-center">
              {['I have a headache', 'My skin has a rash', 'Chest pain when walking'].map((suggestion) => (
                <button
                  key={suggestion}
                  onClick={() => setInput(suggestion)}
                  className="px-3 py-1.5 bg-white border rounded-full text-sm text-gray-600 hover:bg-blue-50 hover:border-blue-300 transition"
                >
                  {suggestion}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((message) => (
            <div key={message.id} className={`flex gap-3 ${message.role === 'user' ? 'flex-row-reverse' : ''}`}>
              <div className={`w-8 h-8 rounded-full flex items-center justify-center text-sm ${
                message.role === 'user' 
                  ? 'bg-blue-600 text-white' 
                  : message.isEmergency 
                    ? 'bg-red-600 text-white' 
                    : 'bg-gray-300 text-gray-600'
              }`}>
                {message.role === 'user' ? '👤' : '🤖'}
              </div>
              <div className={`max-w-[70%] ${message.role === 'user' ? 'items-end' : 'items-start'}`}>
                {message.agent && (
                  <span className={`text-xs px-2 py-0.5 rounded-full mb-1 inline-block ${agentColors[message.agent] || 'bg-gray-100'}`}>
                    {message.agent}
                  </span>
                )}
                <div className={`px-4 py-2 rounded-2xl ${
                  message.role === 'user'
                    ? 'bg-blue-600 text-white rounded-br-sm'
                    : message.isEmergency
                      ? 'bg-red-50 border-2 border-red-400 text-red-900 rounded-bl-sm'
                      : 'bg-white border text-gray-800 rounded-bl-sm'
                }`}>
                  <div className="whitespace-pre-wrap text-sm">{message.content}</div>
                </div>
                <div className="text-xs text-gray-400 mt-1 px-1">
                  {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                </div>
              </div>
            </div>
          ))
        )}
        
        {isLoading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 rounded-full bg-gray-300 flex items-center justify-center">🤖</div>
            <div className="bg-white border rounded-2xl rounded-bl-sm px-4 py-3">
              <div className="flex gap-1">
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '150ms' }} />
                <span className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" style={{ animationDelay: '300ms' }} />
              </div>
            </div>
          </div>
        )}
        
        <div ref={messagesEndRef} />
      </div>

      {/* Image Preview */}
      {selectedImage && (
        <div className="bg-gray-100 border-t px-4 py-2 flex items-center gap-3">
          <img src={`data:image/jpeg;base64,${selectedImage}`} alt="Preview" className="w-12 h-12 object-cover rounded" />
          <span className="text-sm text-gray-600">📷 Image ready</span>
          <button onClick={() => setSelectedImage(null)} className="ml-auto text-gray-500 hover:text-red-500">✕</button>
        </div>
      )}

      {/* Input */}
      <div className="bg-white border-t p-4">
        <div className="flex items-end gap-2">
          <input type="file" ref={fileInputRef} onChange={handleImageSelect} accept="image/*" className="hidden" />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="p-2.5 text-gray-500 hover:text-blue-600 hover:bg-blue-50 rounded-lg transition"
            title="Upload image"
          >
            📷
          </button>
          <textarea
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={handleKeyPress}
            placeholder="Describe your symptoms..."
            rows={1}
            className="flex-1 px-4 py-2.5 border rounded-xl resize-none focus:outline-none focus:ring-2 focus:ring-blue-500"
          />
          <button
            onClick={handleSend}
            disabled={isLoading || (!input.trim() && !selectedImage)}
            className={`p-2.5 rounded-xl transition ${
              isLoading || (!input.trim() && !selectedImage)
                ? 'bg-gray-200 text-gray-400'
                : 'bg-blue-600 text-white hover:bg-blue-700'
            }`}
          >
            ➤
          </button>
        </div>
        <p className="text-xs text-gray-400 mt-2 text-center">
          ⚕️ For informational purposes only. Always consult a healthcare professional.
        </p>
      </div>
    </div>
  );
}

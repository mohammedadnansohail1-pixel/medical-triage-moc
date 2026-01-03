import { Message } from '../types';
import { User, Bot, AlertTriangle, Heart, Stethoscope, Activity } from 'lucide-react';

interface ChatMessageProps {
  message: Message;
}

const agentIcons: Record<string, React.ReactNode> = {
  emergency: <AlertTriangle className="w-4 h-4" />,
  cardiology: <Heart className="w-4 h-4" />,
  dermatology: <Stethoscope className="w-4 h-4" />,
  supervisor: <Activity className="w-4 h-4" />,
  triage: <Stethoscope className="w-4 h-4" />,
};

const agentColors: Record<string, string> = {
  emergency: 'bg-red-100 text-red-800 border-red-300',
  cardiology: 'bg-pink-100 text-pink-800 border-pink-300',
  dermatology: 'bg-purple-100 text-purple-800 border-purple-300',
  supervisor: 'bg-blue-100 text-blue-800 border-blue-300',
  triage: 'bg-green-100 text-green-800 border-green-300',
};

export function ChatMessage({ message }: ChatMessageProps) {
  const isUser = message.role === 'user';
  const isEmergency = message.isEmergency;

  return (
    <div className={`flex gap-3 ${isUser ? 'flex-row-reverse' : ''} ${isEmergency ? 'animate-pulse' : ''}`}>
      {/* Avatar */}
      <div className={`flex-shrink-0 w-9 h-9 rounded-full flex items-center justify-center ${
        isUser 
          ? 'bg-blue-600 text-white' 
          : isEmergency 
            ? 'bg-red-600 text-white' 
            : 'bg-gray-200 text-gray-600'
      }`}>
        {isUser ? <User className="w-5 h-5" /> : <Bot className="w-5 h-5" />}
      </div>

      {/* Message content */}
      <div className={`flex flex-col max-w-[75%] ${isUser ? 'items-end' : 'items-start'}`}>
        {/* Agent badge */}
        {!isUser && message.agent && (
          <div className={`flex items-center gap-1.5 px-2 py-0.5 rounded-full text-xs font-medium mb-1 border ${
            agentColors[message.agent] || 'bg-gray-100 text-gray-800 border-gray-300'
          }`}>
            {agentIcons[message.agent]}
            <span className="capitalize">{message.agent}</span>
          </div>
        )}

        {/* Message bubble */}
        <div className={`px-4 py-2.5 rounded-2xl ${
          isUser 
            ? 'bg-blue-600 text-white rounded-br-md' 
            : isEmergency
              ? 'bg-red-50 text-red-900 border-2 border-red-400 rounded-bl-md'
              : 'bg-gray-100 text-gray-900 rounded-bl-md'
        }`}>
          <div className="whitespace-pre-wrap text-sm leading-relaxed">
            {message.content}
          </div>
        </div>

        {/* Timestamp */}
        <span className="text-xs text-gray-400 mt-1 px-1">
          {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
        </span>
      </div>
    </div>
  );
}

"use client";

import { useState, useEffect, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { 
  Send, 
  Search, 
  FileText, 
  ChevronRight, 
  ShieldCheck, 
  RefreshCcw,
  PlusCircle,
  BarChart3,
  Cpu
} from "lucide-react";

interface Message {
  role: "user" | "assistant";
  content: string;
  thought?: string;
  calculation?: string;
  sources?: any[];
  expandedSource?: number | null;
}

export default function Home() {
  const [input, setInput] = useState("");
  const [messages, setMessages] = useState<Message[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [companies, setCompanies] = useState<string[]>([]);
  const [selectedCompany, setSelectedCompany] = useState("全部公司");
  const [topK, setTopK] = useState(5);
  const scrollRef = useRef<HTMLDivElement>(null);

  // 初始化获取公司列表
  useEffect(() => {
    fetch("http://localhost:8000/api/companies")
      .then(res => res.json())
      .then(data => setCompanies(data))
      .catch(err => console.error("Failed to fetch companies", err));
  }, []);

  // 自动滚动到底部
  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isLoading) return;

    const userMsg: Message = { role: "user", content: input };
    setMessages(prev => [...prev, userMsg]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch("http://localhost:8000/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          prompt: input,
          companies: [selectedCompany],
          top_k: topK
        })
      });

      const data = await response.json();
      const assistantMsg: Message = {
        role: "assistant",
        content: data.answer,
        thought: data.thought,
        calculation: data.calculation,
        sources: data.sources
      };
      setMessages(prev => [...prev, assistantMsg]);
    } catch (error) {
      console.error("Chat error:", error);
      setMessages(prev => [...prev, { role: "assistant", content: "❌ 服务连接失败，请确保 FastAPI 后端已启动。" }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex h-screen bg-apple-gray text-apple-dark overflow-hidden font-sans">
      {/* 侧边栏 - 极致简约 */}
      <aside className="w-72 bg-white border-r border-gray-200 flex flex-col p-6 hidden md:flex">
        <div className="flex items-center gap-2 mb-10 px-2">
          <div className="w-8 h-8 bg-black rounded-lg flex items-center justify-center">
            <BarChart3 className="text-white w-5 h-5" />
          </div>
          <h1 className="font-semibold text-lg tracking-tight">Financial Insight</h1>
        </div>

        <div className="space-y-6 flex-1">
          {/* Top-K Slider */}
          <div className="px-2">
            <div className="flex justify-between items-center mb-3">
              <label className="text-xs font-medium text-gray-400 uppercase tracking-widest">
                Search Depth (K)
              </label>
              <span className="text-xs font-bold text-apple-blue bg-blue-50 px-2 py-0.5 rounded-full">
                {topK}
              </span>
            </div>
            <input 
              type="range" 
              min="1" 
              max="15" 
              value={topK} 
              onChange={(e) => setTopK(parseInt(e.target.value))}
              className="w-full h-1.5 bg-gray-100 rounded-lg appearance-none cursor-pointer accent-apple-blue"
            />
            <div className="flex justify-between text-[10px] text-gray-300 mt-2 font-medium">
              <span>Fast</span>
              <span>Precise</span>
            </div>
          </div>

          <div>
            <label className="text-xs font-medium text-gray-400 uppercase tracking-widest mb-3 block px-2">
              Data Scope
            </label>
            <div className="space-y-1 overflow-y-auto max-h-[60vh]">
              {["全部公司", ...companies].map((company) => (
                <button
                  key={company}
                  onClick={() => setSelectedCompany(company)}
                  className={`w-full text-left px-3 py-2 rounded-xl text-sm transition-all duration-200 ${
                    selectedCompany === company 
                    ? "bg-apple-gray font-medium" 
                    : "text-gray-500 hover:bg-gray-50"
                  }`}
                >
                  {company.length > 20 ? company.substring(0, 18) + "..." : company}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="mt-auto p-4 bg-gray-50 rounded-2xl border border-gray-100">
          <div className="flex items-center gap-2 mb-2">
            <ShieldCheck className="w-4 h-4 text-green-500" />
            <span className="text-xs font-medium">Enterprise Security</span>
          </div>
          <p className="text-[10px] text-gray-400">Local Agent active. Python verified calculations enabled.</p>
        </div>
      </aside>

      {/* 主界面 */}
      <section className="flex-1 flex flex-col relative bg-white md:rounded-l-[40px] shadow-2xl overflow-hidden">
        {/* 顶部导航 */}
        <header className="h-20 glass sticky top-0 z-10 flex items-center justify-between px-8 border-b border-gray-100">
          <div className="flex flex-col">
            <span className="text-xs font-semibold text-apple-blue uppercase tracking-tighter">Agentic Financial RAG V2</span>
            <span className="text-sm font-medium text-gray-500">{selectedCompany}</span>
          </div>
          <div className="flex items-center gap-4">
            <button 
              onClick={() => setMessages([])}
              className="p-2 hover:bg-gray-100 rounded-full transition-colors"
            >
              <RefreshCcw className="w-5 h-5 text-gray-400" />
            </button>
          </div>
        </header>

        {/* 聊天区域 */}
        <div 
          ref={scrollRef}
          className="flex-1 overflow-y-auto px-6 py-10 space-y-8 scroll-smooth"
        >
          {messages.length === 0 && (
            <div className="h-full flex flex-col items-center justify-center text-center max-w-md mx-auto space-y-6">
              <motion.div 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                className="w-16 h-16 bg-apple-gray rounded-3xl flex items-center justify-center mb-4"
              >
                <Cpu className="w-8 h-8 text-apple-blue" />
              </motion.div>
              <motion.h2 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.1 }}
                className="text-3xl font-bold tracking-tight"
              >
                AI Financial Agent
              </motion.h2>
              <motion.p 
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ delay: 0.2 }}
                className="text-gray-400 text-sm leading-relaxed"
              >
                Analyzing reports with precise logic. I can now perform verified Python calculations for 100% accuracy.
              </motion.p>
            </div>
          )}

          <AnimatePresence mode="popLayout">
            {messages.map((msg, index) => (
              <motion.div
                key={index}
                initial={{ opacity: 0, y: 20, scale: 0.95 }}
                animate={{ opacity: 1, y: 0, scale: 1 }}
                transition={{ duration: 0.4, ease: [0.23, 1, 0.32, 1] }}
                className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
              >
                <div className={`max-w-[85%] p-5 rounded-[24px] ${
                  msg.role === "user" 
                  ? "bg-apple-blue text-white shadow-lg shadow-blue-100" 
                  : "bg-apple-gray text-apple-dark"
                }`}>
                  
                  {/* Analysis Log (Agent Thought) */}
                  {msg.role === "assistant" && (msg.thought || msg.calculation) && (
                    <div className="mb-4 bg-white/50 border border-white/30 p-4 rounded-2xl shadow-sm">
                      <div className="flex items-center gap-2 mb-3 text-apple-blue font-bold text-[10px] uppercase tracking-widest">
                        <Search className="w-3.5 h-3.5" />
                        Analysis Log
                      </div>
                      {msg.thought && (
                        <div className="text-[12px] text-gray-500 italic mb-3 leading-relaxed border-l-2 border-apple-blue/20 pl-3">
                          {msg.thought}
                        </div>
                      )}
                      {msg.calculation && (
                        <div className="bg-black/5 p-3 rounded-xl font-mono text-[11px] text-apple-blue border border-black/5">
                          <div className="flex items-center gap-1.5 mb-1.5 opacity-60">
                            <Cpu className="w-3 h-3" />
                            <span>Python Verification</span>
                          </div>
                          <code className="block whitespace-pre-wrap">{msg.calculation}</code>
                        </div>
                      )}
                    </div>
                  )}

                  <p className="text-[15px] leading-relaxed whitespace-pre-wrap font-medium">{msg.content}</p>
                  
                  {msg.sources && msg.sources.length > 0 && (
                    <div className="mt-5 pt-5 border-t border-gray-200/50 space-y-3">
                      <span className="text-[10px] font-bold uppercase tracking-widest text-gray-400">Sources & Evidence</span>
                      <div className="grid grid-cols-1 gap-2.5">
                        {msg.sources.map((src, i) => (
                          <div key={i} className="flex flex-col gap-1.5">
                            <div 
                              onClick={() => {
                                const updatedMessages = [...messages];
                                const currentMsg = updatedMessages[index];
                                if (currentMsg.expandedSource === i) {
                                  currentMsg.expandedSource = null;
                                } else {
                                  currentMsg.expandedSource = i;
                                }
                                setMessages(updatedMessages);
                              }}
                              className="bg-white/60 p-3.5 rounded-xl flex items-center justify-between group cursor-pointer hover:bg-white transition-all active:scale-[0.98]"
                            >
                              <div className="flex items-center gap-3">
                                <FileText className="w-4 h-4 text-apple-blue" />
                                <div className="flex flex-col">
                                  <span className="text-[11px] font-semibold truncate max-w-[200px]">{src.filename || "Source"}</span>
                                  <span className="text-[9px] text-gray-400 uppercase tracking-tighter font-bold">Page {src.page || "?"}</span>
                                </div>
                              </div>
                              <ChevronRight className={`w-4 h-4 text-gray-300 group-hover:text-apple-blue transition-transform duration-300 ${
                                msg.expandedSource === i ? "rotate-90" : ""
                              }`} />
                            </div>
                            
                            {msg.expandedSource === i && (
                              <motion.div 
                                initial={{ opacity: 0, height: 0, y: -10 }}
                                animate={{ opacity: 1, height: "auto", y: 0 }}
                                className="bg-white/90 p-4 rounded-xl text-[12px] text-gray-600 leading-relaxed border border-gray-100 shadow-inner overflow-hidden font-normal"
                              >
                                {src.content}
                              </motion.div>
                            )}
                          </div>
                        ))}
                      </div>
                    </div>
                  )}
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
          
          {isLoading && (
            <motion.div 
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className="flex justify-start"
            >
              <div className="bg-apple-gray p-5 rounded-[24px] flex items-center gap-2">
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.2s]" />
                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce [animation-delay:0.4s]" />
              </div>
            </motion.div>
          )}
        </div>

        {/* 输入框区域 */}
        <div className="p-8">
          <div className="max-w-3xl mx-auto relative flex items-center">
            <div className="absolute left-5 text-gray-400">
              <PlusCircle className="w-5 h-5 hover:text-apple-blue cursor-pointer transition-colors" />
            </div>
            <input 
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && handleSend()}
              placeholder="Ask for calculations or comparisons (e.g. Compare BYD and Vanke's revenue)..."
              className="w-full bg-apple-gray rounded-[28px] py-5 pl-14 pr-16 focus:outline-none focus:ring-2 focus:ring-blue-100 transition-all text-[15px] placeholder:text-gray-400 shadow-sm"
            />
            <button 
              onClick={handleSend}
              disabled={!input.trim() || isLoading}
              className={`absolute right-3 w-10 h-10 rounded-full flex items-center justify-center transition-all ${
                input.trim() && !isLoading 
                ? "bg-apple-blue text-white shadow-md shadow-blue-200 scale-100" 
                : "bg-gray-200 text-white scale-90"
              }`}
            >
              <Send className="w-5 h-5" />
            </button>
          </div>
          <p className="text-center text-[10px] text-gray-300 mt-4 uppercase tracking-widest font-bold">
            Verified Analysis Engine Active
          </p>
        </div>
      </section>
    </main>
  );
}

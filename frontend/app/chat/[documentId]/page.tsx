'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter, useParams } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import { apiClient, type Document, type Session, type Message } from '@/lib/api'

export default function ChatPage() {
  const params = useParams()
  const documentId = params.documentId as string
  const router = useRouter()

  const [userId, setUserId] = useState<string | null>(null)
  const [document, setDocument] = useState<Document | null>(null)
  const [sessions, setSessions] = useState<Session[]>([])
  const [activeSessionId, setActiveSessionId] = useState<string | null>(null)
  const [messages, setMessages] = useState<Message[]>([])
  const [input, setInput] = useState('')
  const [isStreaming, setIsStreaming] = useState(false)
  const [streamingContent, setStreamingContent] = useState('')
  const [loading, setLoading] = useState(true)
  const messagesEndRef = useRef<HTMLDivElement>(null)

  useEffect(() => {
    const supabase = createClient()
    supabase.auth.getUser().then(async ({ data: { user } }) => {
      if (!user) {
        router.replace('/login')
        return
      }
      setUserId(user.id)
      const api = apiClient(user.id)

      const [docs, allSessions] = await Promise.all([
        api.listDocuments(),
        api.listSessions(),
      ])

      const doc = docs.find((d) => d.id === documentId) ?? null
      setDocument(doc)

      const docSessions = allSessions.filter((s) => s.document_id === documentId)
      setSessions(docSessions)

      if (docSessions.length > 0) {
        const latest = docSessions[0]
        setActiveSessionId(latest.id)
        const msgs = await api.getMessages(latest.id)
        setMessages(msgs)
      }

      setLoading(false)
    })
  }, [documentId])

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages, streamingContent])

  async function handleNewSession() {
    if (!userId) return
    const session = await apiClient(userId).createSession(documentId)
    setSessions((prev) => [session, ...prev])
    setActiveSessionId(session.id)
    setMessages([])
  }

  async function handleSelectSession(id: string) {
    if (!userId || id === activeSessionId) return
    setActiveSessionId(id)
    const msgs = await apiClient(userId).getMessages(id)
    setMessages(msgs)
  }

  async function handleSend(e: React.FormEvent) {
    e.preventDefault()
    if (!input.trim() || !userId || isStreaming) return

    let sessionId = activeSessionId
    if (!sessionId) {
      const session = await apiClient(userId).createSession(documentId)
      setSessions((prev) => [session, ...prev])
      setActiveSessionId(session.id)
      sessionId = session.id
    }

    const userMsg: Message = {
      id: crypto.randomUUID(),
      role: 'user',
      content: input.trim(),
      created_at: new Date().toISOString(),
    }
    setMessages((prev) => [...prev, userMsg])
    setInput('')
    setIsStreaming(true)
    setStreamingContent('')

    apiClient(userId).streamMessage(
      sessionId,
      userMsg.content,
      documentId,
      (token) => setStreamingContent((prev) => prev + token),
      (fullContent) => {
        setIsStreaming(false)
        setMessages((prev) => [
          ...prev,
          {
            id: crypto.randomUUID(),
            role: 'assistant',
            content: fullContent,
            created_at: new Date().toISOString(),
          },
        ])
        setStreamingContent('')
      },
      () => {
        setIsStreaming(false)
        setStreamingContent('')
      }
    )
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    )
  }

  return (
    <div className="h-screen flex overflow-hidden">
      {/* Sidebar */}
      <aside className="w-56 border-r border-border flex flex-col shrink-0">
        <div className="p-4 border-b border-border">
          <button
            onClick={() => router.push('/documents')}
            className="text-xs text-muted-foreground hover:text-foreground transition-colors"
          >
            ← Documents
          </button>
          <p className="mt-2 text-sm font-medium truncate" title={document?.filename}>
            {document?.filename ?? 'Document'}
          </p>
        </div>

        <div className="flex-1 overflow-y-auto p-2 space-y-0.5">
          {sessions.map((s) => (
            <button
              key={s.id}
              onClick={() => handleSelectSession(s.id)}
              className={`w-full text-left px-3 py-2 rounded-md text-xs transition-colors ${
                s.id === activeSessionId
                  ? 'bg-muted font-medium text-foreground'
                  : 'text-muted-foreground hover:bg-muted/50 hover:text-foreground'
              }`}
            >
              {s.title ?? new Date(s.created_at).toLocaleDateString()}
            </button>
          ))}
          {sessions.length === 0 && (
            <p className="text-xs text-muted-foreground px-3 py-2">No sessions yet</p>
          )}
        </div>

        <div className="p-4 border-t border-border">
          <button
            onClick={handleNewSession}
            className="w-full h-8 text-xs font-medium rounded-md border border-border hover:bg-muted transition-colors"
          >
            + New Chat
          </button>
        </div>
      </aside>

      {/* Main */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {activeSessionId ? (
          <>
            <div className="flex-1 overflow-y-auto px-6 py-6 space-y-4">
              {messages.length === 0 && !isStreaming && (
                <p className="text-sm text-muted-foreground text-center pt-16">
                  Ask anything about this document.
                </p>
              )}

              {messages.map((msg) => (
                <div
                  key={msg.id}
                  className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                >
                  <div
                    className={`max-w-[72%] rounded-lg px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                      msg.role === 'user'
                        ? 'bg-primary text-primary-foreground'
                        : 'bg-muted border border-border text-foreground'
                    }`}
                  >
                    {msg.content}
                  </div>
                </div>
              ))}

              {isStreaming && (
                <div className="flex justify-start">
                  <div className="max-w-[72%] rounded-lg px-4 py-3 text-sm leading-relaxed bg-muted border border-border text-foreground whitespace-pre-wrap">
                    {streamingContent || (
                      <span className="text-muted-foreground">Thinking…</span>
                    )}
                    {streamingContent && (
                      <span className="inline-block w-0.5 h-3.5 ml-0.5 bg-foreground/40 animate-pulse align-middle" />
                    )}
                  </div>
                </div>
              )}

              <div ref={messagesEndRef} />
            </div>

            <div className="border-t border-border p-4">
              <form onSubmit={handleSend} className="flex gap-2">
                <input
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  disabled={isStreaming}
                  placeholder="Ask about this document…"
                  className="flex-1 h-9 px-3 text-sm rounded-md border border-input bg-transparent focus:outline-none focus:ring-1 focus:ring-ring disabled:opacity-50"
                />
                <button
                  type="submit"
                  disabled={isStreaming || !input.trim()}
                  className="h-9 px-4 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 disabled:opacity-50 transition-colors"
                >
                  Send
                </button>
              </form>
            </div>
          </>
        ) : (
          <div className="flex-1 flex items-center justify-center">
            <div className="text-center">
              <p className="text-sm text-muted-foreground">No active session</p>
              <button
                onClick={handleNewSession}
                className="mt-3 h-9 px-4 text-sm font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
              >
                Start Chat
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}

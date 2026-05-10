const API_URL = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

export interface Document {
  id: string
  filename: string
  doc_type: string | null
  chunk_count: number
  created_at: string
}

export interface Session {
  id: string
  document_id: string
  title: string | null
  created_at: string
  updated_at: string
}

export interface Message {
  id: string
  role: 'user' | 'assistant'
  content: string
  created_at: string
}

export function apiClient(userId: string) {
  const headers: Record<string, string> = { 'X-User-Id': userId }

  return {
    async listDocuments(): Promise<Document[]> {
      const res = await fetch(`${API_URL}/documents/`, { headers })
      if (!res.ok) throw new Error('Failed to load documents')
      return res.json()
    },

    async uploadDocument(file: File): Promise<Document> {
      const form = new FormData()
      form.append('file', file)
      const res = await fetch(`${API_URL}/documents/upload`, {
        method: 'POST',
        headers,
        body: form,
      })
      if (!res.ok) {
        const err = await res.json().catch(() => ({ detail: 'Upload failed' }))
        throw new Error(err.detail ?? 'Upload failed')
      }
      return res.json()
    },

    async deleteDocument(id: string): Promise<void> {
      await fetch(`${API_URL}/documents/${id}`, { method: 'DELETE', headers })
    },

    async createSession(documentId: string): Promise<Session> {
      const res = await fetch(
        `${API_URL}/chat/sessions?document_id=${documentId}`,
        { method: 'POST', headers }
      )
      if (!res.ok) throw new Error('Failed to create session')
      return res.json()
    },

    async listSessions(): Promise<Session[]> {
      const res = await fetch(`${API_URL}/chat/sessions`, { headers })
      if (!res.ok) throw new Error('Failed to load sessions')
      return res.json()
    },

    async getMessages(sessionId: string): Promise<Message[]> {
      const res = await fetch(
        `${API_URL}/chat/sessions/${sessionId}/messages`,
        { headers }
      )
      if (!res.ok) throw new Error('Failed to load messages')
      return res.json()
    },

    streamMessage(
      sessionId: string,
      query: string,
      documentId: string,
      onToken: (token: string) => void,
      onDone: (fullContent: string) => void,
      onError: (err: string) => void
    ): void {
      fetch(`${API_URL}/chat/sessions/${sessionId}/messages`, {
        method: 'POST',
        headers: { ...headers, 'Content-Type': 'application/json' },
        body: JSON.stringify({ query, document_id: documentId }),
      })
        .then(async (res) => {
          if (!res.ok || !res.body) {
            onError('Request failed')
            return
          }
          const reader = res.body.getReader()
          const decoder = new TextDecoder()
          let buffer = ''
          let fullContent = ''

          while (true) {
            const { done, value } = await reader.read()
            if (done) break
            buffer += decoder.decode(value, { stream: true })
            const lines = buffer.split('\n')
            buffer = lines.pop() ?? ''

            for (const line of lines) {
              if (!line.startsWith('data: ')) continue
              const data = line.slice(6).trim()
              if (data === '[DONE]') {
                onDone(fullContent)
                return
              }
              try {
                const parsed = JSON.parse(data)
                if (parsed.token) {
                  fullContent += parsed.token
                  onToken(parsed.token)
                }
              } catch {}
            }
          }
          onDone(fullContent)
        })
        .catch((err: Error) => onError(err.message))
    },
  }
}

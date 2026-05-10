'use client'

import { useEffect, useRef, useState } from 'react'
import { useRouter } from 'next/navigation'
import { createClient } from '@/lib/supabase/client'
import { apiClient, type Document } from '@/lib/api'

const TYPE_LABELS: Record<string, string> = {
  legal: 'Legal',
  academic: 'Academic',
  financial: 'Financial',
  technical: 'Technical',
  general: 'General',
}

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
  })
}

export default function DocumentsPage() {
  const [documents, setDocuments] = useState<Document[]>([])
  const [loading, setLoading] = useState(true)
  const [uploading, setUploading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [userId, setUserId] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const router = useRouter()

  useEffect(() => {
    const supabase = createClient()
    supabase.auth.getUser().then(({ data: { user } }) => {
      if (!user) {
        router.replace('/login')
        return
      }
      setUserId(user.id)
      apiClient(user.id)
        .listDocuments()
        .then(setDocuments)
        .catch((e: Error) => setError(e.message))
        .finally(() => setLoading(false))
    })
  }, [])

  async function handleUpload(e: React.ChangeEvent<HTMLInputElement>) {
    const file = e.target.files?.[0]
    if (!file || !userId) return
    setUploading(true)
    setError(null)
    try {
      const doc = await apiClient(userId).uploadDocument(file)
      setDocuments((prev) => [doc, ...prev])
    } catch (err: unknown) {
      setError((err as Error).message)
    } finally {
      setUploading(false)
      if (fileInputRef.current) fileInputRef.current.value = ''
    }
  }

  async function handleDelete(id: string) {
    if (!userId) return
    await apiClient(userId).deleteDocument(id)
    setDocuments((prev) => prev.filter((d) => d.id !== id))
  }

  async function handleSignOut() {
    const supabase = createClient()
    await supabase.auth.signOut()
    router.push('/login')
  }

  if (loading) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <p className="text-sm text-muted-foreground">Loading…</p>
      </div>
    )
  }

  return (
    <div className="min-h-screen bg-background">
      <header className="border-b border-border">
        <div className="max-w-5xl mx-auto px-6 h-14 flex items-center justify-between">
          <span className="text-sm font-semibold">Mr.Summarizer</span>
          <div className="flex items-center gap-3">
            <input
              ref={fileInputRef}
              type="file"
              accept=".pdf"
              className="hidden"
              onChange={handleUpload}
            />
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploading}
              className="h-8 px-3 text-xs font-medium rounded-md border border-border hover:bg-muted disabled:opacity-50 transition-colors"
            >
              {uploading ? 'Uploading…' : 'Upload PDF'}
            </button>
            <button
              onClick={handleSignOut}
              className="h-8 px-3 text-xs font-medium text-muted-foreground hover:text-foreground transition-colors"
            >
              Sign out
            </button>
          </div>
        </div>
      </header>

      <main className="max-w-5xl mx-auto px-6 py-8">
        {error && (
          <div className="mb-6 px-4 py-3 rounded-md border border-destructive/30 bg-destructive/5 text-sm text-destructive">
            {error}
          </div>
        )}

        {documents.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-24 text-center">
            <p className="text-sm font-medium">No documents yet</p>
            <p className="mt-1 text-sm text-muted-foreground">
              Upload a PDF to get started
            </p>
          </div>
        ) : (
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="rounded-lg border border-border p-4 flex flex-col gap-3"
              >
                <div className="flex items-start justify-between gap-2">
                  <p className="text-sm font-medium leading-snug break-all line-clamp-2">
                    {doc.filename}
                  </p>
                  {doc.doc_type && (
                    <span className="shrink-0 text-xs px-2 py-0.5 rounded-full border border-border text-muted-foreground">
                      {TYPE_LABELS[doc.doc_type] ?? doc.doc_type}
                    </span>
                  )}
                </div>

                <p className="text-xs text-muted-foreground">
                  {doc.chunk_count} chunks · {formatDate(doc.created_at)}
                </p>

                <div className="flex gap-2 mt-auto pt-1">
                  <button
                    onClick={() => router.push(`/chat/${doc.id}`)}
                    className="flex-1 h-8 text-xs font-medium rounded-md bg-primary text-primary-foreground hover:bg-primary/90 transition-colors"
                  >
                    Open Chat
                  </button>
                  <button
                    onClick={() => handleDelete(doc.id)}
                    className="h-8 px-3 text-xs font-medium rounded-md border border-border text-muted-foreground hover:text-destructive hover:border-destructive/40 transition-colors"
                  >
                    Delete
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  )
}

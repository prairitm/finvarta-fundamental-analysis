import { useState } from 'react'
import DOMPurify from 'dompurify'
import { marked } from 'marked'
import './App.css'

const API_URL = 'http://localhost:8000/analyze'

const humanizeKey = (key) =>
  key
    .replace(/[_-]/g, ' ')
    .replace(/\b\w/g, (char) => char.toUpperCase())

const stringifyValue = (value) => {
  if (value === null || value === undefined) return ''
  if (typeof value === 'string') return value
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (Array.isArray(value)) {
    return value
      .map((item, index) => `• ${Array.isArray(value) ? '' : ''}${stringifyValue(item) || `Item ${index + 1}`}`)
      .join('\n')
  }
  if (typeof value === 'object') {
    return Object.entries(value)
      .map(([innerKey, innerValue]) => `${humanizeKey(innerKey)}: ${stringifyValue(innerValue)}`)
      .join('\n')
  }
  return ''
}

const toSections = (payload) => {
  if (!payload) return []
  if (typeof payload === 'string') {
    return payload
      .split(/\n{2,}/)
      .map((block) => block.trim())
      .filter(Boolean)
      .map((block) => ({ title: null, content: block }))
  }
  if (Array.isArray(payload)) {
    return payload.map((entry, index) => ({
      title: `Insight ${index + 1}`,
      content: stringifyValue(entry),
    }))
  }
  if (typeof payload === 'object') {
    return Object.entries(payload).map(([key, value]) => ({
      title: humanizeKey(key),
      content: stringifyValue(value),
    }))
  }
  return [{ title: null, content: String(payload) }]
}

function App() {
  const [company, setCompany] = useState('')
  const [sections, setSections] = useState([])
  const [meta, setMeta] = useState({})
  const [error, setError] = useState('')
  const [isLoading, setIsLoading] = useState(false)

  const handleSubmit = async (event) => {
    event.preventDefault()
    if (!company.trim()) {
      setError('Please enter a company name')
      return
    }

    setIsLoading(true)
    setError('')
    setSections([])
    setMeta({})

    try {
      const response = await fetch(API_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ company: company.trim() }),
      })

      if (!response.ok) {
        const message = await response.text()
        throw new Error(message || 'Analysis failed')
      }

      const data = await response.json()
      setSections(toSections(data?.analysis || data))
      if (data && typeof data === 'object') {
        const { analysis, ...rest } = data
        setMeta(rest)
      }
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setIsLoading(false)
    }
  }

  return (
    <main className="app">
      <section className="panel">
        <div className="panel-header">
          <div>
            <h1>Finvarta Analysis</h1>
            <p className="description">
              Submit a company name to run the fundamental analysis pipeline.
            </p>
          </div>
          <div className="badge">FastAPI · React</div>
        </div>

        <form className="form" onSubmit={handleSubmit}>
          <label htmlFor="company">Company name</label>
          <div className="input-row">
            <input
              id="company"
              type="text"
              placeholder="Acme Corp"
              value={company}
              onChange={(event) => setCompany(event.target.value)}
              disabled={isLoading}
            />
            <button type="submit" disabled={isLoading || !company.trim()}>
              {isLoading ? 'Fetching…' : 'Submit'}
            </button>
          </div>
        </form>
        {error && <p className="status error">{error}</p>}

        <section className="analysis-block">
          <header>
            <h2>Analysis Overview</h2>
            {meta?.model && <span className="badge subtle">Model: {meta.model}</span>}
          </header>

          {sections.length > 0 ? (
            <div className="reader">
              {sections.map((section, index) => (
                <article key={`${section.title ?? 'section'}-${index}`}>
                  {section.title && <h3>{section.title}</h3>}
                  <div
                    className="markdown"
                    dangerouslySetInnerHTML={{
                      __html: DOMPurify.sanitize(
                        marked.parse(section.content || '', { breaks: true }),
                      ),
                    }}
                  />
                </article>
              ))}
            </div>
          ) : (
            <div className="placeholder card">
              <p>Results will appear here after you submit a request.</p>
              <small>Tip: make sure the FastAPI server is running on port 8000.</small>
            </div>
          )}
        </section>
      </section>
    </main>
  )
}

export default App

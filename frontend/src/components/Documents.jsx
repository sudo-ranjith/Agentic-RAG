import { useState } from 'react'

const API_URL = 'http://localhost:8000'

export default function Documents() {
  const [isUploading, setIsUploading] = useState(false)
  const [message, setMessage] = useState('')

  const handleUpload = async (e) => {
    const files = e.target.files
    if (!files.length) return

    const formData = new FormData()
    Array.from(files).forEach(file => {
      formData.append('files', file)
    })

    try {
      setIsUploading(true)
      setMessage('Uploading...')

      const response = await fetch(`${API_URL}/upload`, {
        method: 'POST',
        body: formData
      })

      if (!response.ok) throw new Error('Upload failed')

      const data = await response.json()
      setMessage(`Successfully uploaded ${data.ingested} documents`)
    } catch (err) {
      console.error('Upload error:', err)
      setMessage('Error uploading files')
    } finally {
      setIsUploading(false)
    }
  }

  return (
    <div className="p-4">
      <h2>Upload Documents</h2>
      <div className="card">
        <input
          type="file"
          multiple
          onChange={handleUpload}
          disabled={isUploading}
        />
        {message && <p className="mt-2">{message}</p>}
      </div>
    </div>
  )
}

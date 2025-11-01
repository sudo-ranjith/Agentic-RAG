// frontend/src/components/TypingDots.jsx
export default function TypingDots() {
  return (
    <div style={{
      display: "inline-flex",
      gap: 6,
      alignItems: "center",
      padding: "10px 12px",
      background: "#ffffff",
      border: "1px solid #e5e7eb",
      borderRadius: 12,
      boxShadow: "0 1px 2px rgba(16,24,40,.05)"
    }}>
      <span style={{ width: 6, height: 6, background: "#9ca3af", borderRadius: "999px", animation: "blink 1s infinite" }} />
      <span style={{ width: 6, height: 6, background: "#9ca3af", borderRadius: "999px", animation: "blink 1s .15s infinite" }} />
      <span style={{ width: 6, height: 6, background: "#9ca3af", borderRadius: "999px", animation: "blink 1s .3s infinite" }} />
      <style>{`
        @keyframes blink {
          0% { opacity: .3; transform: translateY(0); }
          50% { opacity: 1; transform: translateY(-2px); }
          100% { opacity: .3; transform: translateY(0); }
        }
      `}</style>
    </div>
  );
}

import { useState, type FormEvent } from "react";
import { useAuth } from "../auth";
import { Mark } from "../components/Shell";

export function Auth() {
  const { signIn, signUp } = useAuth();
  const [mode, setMode] = useState<"in" | "up">("in");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const creating = mode === "up";
  const tooShort = creating && password.length > 0 && password.length < 12;

  async function submit(e: FormEvent) {
    e.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await (creating ? signUp : signIn)(email.trim().toLowerCase(), password);
    } catch (err) {
      setError(err instanceof Error ? err.message : "Something went wrong.");
      setBusy(false);
    }
  }

  return (
    <main className="auth">
      <section className="auth-intro">
        <span className="brand"><Mark size={24} /> <span>AIMD Sentinel</span></span>
        <h1>Check whether a photo or video has been edited</h1>
        <p className="lede">
          Drop in a file and get a plain answer in seconds, with the areas that raised a flag marked
          on the image and every check that ran listed beside it.
        </p>
        <ul className="auth-points">
          <li><span className="point-mark" aria-hidden="true" />Your files are visible only to your account</li>
          <li><span className="point-mark" aria-hidden="true" />Every result says which checks ran and which could not</li>
          <li><span className="point-mark" aria-hidden="true" />Delete a file and its results whenever you want</li>
        </ul>
      </section>

      <form className="card auth-form" onSubmit={submit} noValidate>
        <div className="segmented" role="tablist" aria-label="Account">
          <button type="button" role="tab" aria-selected={!creating} onClick={() => { setMode("in"); setError(null); }}>Sign in</button>
          <button type="button" role="tab" aria-selected={creating} onClick={() => { setMode("up"); setError(null); }}>Create account</button>
        </div>
        <div className="field">
          <label htmlFor="email">Email</label>
          <input id="email" className="input" type="email" autoComplete="email" value={email}
            onChange={(e) => setEmail(e.target.value)} required />
        </div>
        <div className="field">
          <label htmlFor="password">Password</label>
          <input id="password" className="input" type="password"
            autoComplete={creating ? "new-password" : "current-password"}
            value={password} onChange={(e) => setPassword(e.target.value)} required
            aria-describedby={creating ? "pw-hint" : undefined} />
          {creating && (
            <span id="pw-hint" className={"hint" + (tooShort ? " warn-text" : "")}>
              At least 12 characters{password ? ` · ${password.length} so far` : ""}
            </span>
          )}
        </div>
        {error && <p className="form-error" role="alert">{error}</p>}
        <button className="btn primary wide" disabled={busy || !email || !password || tooShort}>
          {busy ? (creating ? "Creating account…" : "Signing in…") : creating ? "Create account" : "Sign in"}
        </button>
      </form>
    </main>
  );
}

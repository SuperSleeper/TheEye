// Sign-in page. Also creates the first account when opened on the TheEye computer itself.
const $ = s => document.querySelector(s);
const form = $('#lg-form'), err = $('#lg-err');
let mode = 'login';

async function post(path, body) {
  const r = await fetch(path, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body) });
  if (r.ok) return;
  let msg = `Error ${r.status}`;
  try { const d = await r.json(); msg = typeof d.detail === 'string' ? d.detail : msg; } catch {}
  throw new Error(msg);
}

(async () => {
  const s = await (await fetch('/api/auth/state')).json();
  if (s.signed_in) { location.replace('/'); return; }
  if (!s.configured) {
    $('#lg-intro').classList.remove('hidden');
    if (!s.can_setup) {
      $('#lg-title').textContent = 'No account yet';
      $('#lg-intro').textContent = 'Create the login on the TheEye computer: open this page there (https://localhost:…), '
        + 'or run ./run.sh --set-password in a terminal. Then reload this page.';
      return;
    }
    mode = 'setup';
    $('#lg-title').textContent = 'Create your login';
    $('#lg-intro').textContent = 'Choose the username and password for this TheEye. At least 8 characters; a long passphrase is best.';
    form.password.autocomplete = 'new-password';
    $('#lg-repeat').classList.remove('hidden'); form.repeat.required = true;
    $('#lg-remember').classList.add('hidden');
    $('#lg-submit').textContent = 'Create login';
  }
  form.classList.remove('hidden');
  form.username.focus();
})();

form.onsubmit = async e => {
  e.preventDefault();
  err.textContent = '';
  const btn = $('#lg-submit'); btn.disabled = true;
  try {
    const body = { username: form.username.value, password: form.password.value };
    if (mode === 'setup') {
      if (form.password.value !== form.repeat.value) throw new Error("The passwords don't match.");
      await post('/api/auth/setup', body);
    } else {
      await post('/api/auth/login', { ...body, remember: form.remember.checked });
    }
    location.replace('/');
  } catch (x) {
    err.textContent = x.message; form.password.value = ''; form.password.focus();
  } finally { btn.disabled = false; }
};

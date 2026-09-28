/**
 * Fill PyPI's pending trusted-publisher form.
 *
 * Twelve forms: six projects on pypi.org and six on test.pypi.org. This fills
 * all five fields, deriving the environment from both the project and the
 * hostname.
 *
 * The environment must differ per project. PyPI's pending publishers are unique
 * on (owner, repository, workflow, environment), so a monorepo publishing six
 * projects from one workflow can register exactly one of them under a shared
 * environment -- the second submission is rejected as a duplicate, which is
 * what happens if you give them all `pypi`. See pypi/warehouse#16920. The
 * release workflow uses matching per-project environments.
 *
 * It never submits. It fills, prints what it filled, and you click Add.
 *
 * Progress is read from the "Pending publishers" table on the page, not from
 * anything this script stores. An earlier version marked a project done the
 * moment it typed the values, and then reported "6/6 filled" when only the one
 * that had actually been submitted existed. The table is the truth.
 *
 * DevTools console only. A bookmarklet cannot work here: PyPI sends
 * `script-src 'self'` with no `unsafe-inline`, and Chromium blocks
 * `javascript:` URLs under such a policy. Console execution is exempt.
 *
 * The first paste into a Chrome or Brave console is refused until you type
 * `allow pasting` and press enter. That is the browser, not this script.
 *
 *     jef.next()            fill the next project not yet registered
 *     jef.fill('jef-sdk')   fill a specific one
 *     jef.status()          what the page says is registered
 *
 * Workflow: run jef.next(), click Add, let the page reload, paste this script
 * again, run jef.next(). Six times, then the same on test.pypi.org.
 */
(() => {
  const PROJECTS = ["jef", "jef-core", "jef-scene", "jef-server", "jef-sdk", "jef-train"];
  const OWNER = "cyber-security-dev-dep-mitake-com-tw";
  const REPOSITORY = "jef";
  // The filename, not the path. `.github/workflows/release.yml` is rejected in
  // a way that does not say which field is wrong.
  const WORKFLOW = "release.yml";

  const host = location.hostname;
  // `pypi-jef-core` on pypi.org, `testpypi-jef-core` on test.pypi.org. The
  // prefix differs because the release workflow routes prereleases to TestPyPI.
  const PREFIX = host.startsWith("test.") ? "testpypi-" : "pypi-";
  const environmentFor = (project) => `${PREFIX}${project}`;

  /** Project names PyPI already lists, from the publisher tables. */
  const registered = () => {
    const names = new Set();
    for (const table of document.querySelectorAll(".table--publisher-list")) {
      for (const row of table.querySelectorAll("tbody tr, tr")) {
        const first = row.querySelector("td");
        if (!first) continue;
        const name = first.textContent.trim();
        if (PROJECTS.includes(name)) names.add(name);
      }
    }
    return names;
  };

  const remaining = () => PROJECTS.filter((p) => !registered().has(p));

  const status = () => {
    const have = registered();
    const left = remaining();
    console.log(
      `%c${host}%c — ${have.size}/${PROJECTS.length} registered (read from the page)\n` +
        PROJECTS.map((p) => `  ${have.has(p) ? "✓" : "·"} ${p}`).join("\n") +
        (left.length
          ? `\n\nNext: jef.next()  →  ${left[0]}`
          : `\n\nAll six registered on ${host}. Now do the other registry.`),
      "font-weight:bold",
      "color:inherit"
    );
    return left;
  };

  const form = document.querySelector("#pending-github-publisher-form");
  if (!form) {
    console.error(
      "%cNo GitHub pending-publisher form on this page.",
      "color:#c00;font-weight:bold",
      document.querySelector('form[action*="/account/login"]')
        ? "\nThis is the login page — the publishing page redirects when signed out."
        : "\nOpen /manage/account/publishing/ and scroll to 'Add a new pending publisher'."
    );
    return;
  }

  const set = (id, value) => {
    const field = form.querySelector(`#${id}`);
    if (!field) throw new Error(`field #${id} is not on this form`);
    field.value = value;
    // Dispatched so client-side validation sees a change rather than an
    // apparently untouched field.
    field.dispatchEvent(new Event("input", { bubbles: true }));
    field.dispatchEvent(new Event("change", { bubbles: true }));
  };

  const fill = (name) => {
    if (registered().has(name)) {
      console.warn(`${name} is already registered on ${host}; nothing to do.`);
      return null;
    }
    if (!PROJECTS.includes(name)) {
      console.warn(`${name} is not one of: ${PROJECTS.join(", ")}`);
    }

    const environment = environmentFor(name);
    set("project_name", name);
    set("owner", OWNER);
    set("repository", REPOSITORY);
    set("workflow_filename", WORKFLOW);
    set("environment", environment);

    console.log(
      `%cFilled ${name}%c — now click Add.\n` +
        `  owner        ${OWNER}\n` +
        `  repository   ${REPOSITORY}\n` +
        `  workflow     ${WORKFLOW}\n` +
        `  environment  ${environment}\n\n` +
        "%cOne at a time.%c Calling next() again without clicking Add just " +
        "overwrites these fields, and nothing is registered.",
      "color:#080;font-weight:bold",
      "color:inherit",
      "color:#a60;font-weight:bold",
      "color:inherit"
    );
    return name;
  };

  window.jef = {
    projects: PROJECTS,
    fill,
    registered: () => [...registered()],
    status,
    next: () => {
      const left = remaining();
      if (!left.length) {
        console.log(`All six are registered on ${host}.`);
        return null;
      }
      return fill(left[0]);
    },
  };

  console.log(
    `%cJEF trusted-publisher helper%c — ${host}, environments ${PREFIX}<project>`,
    "color:#2d5f8a;font-weight:bold",
    "color:inherit"
  );
  status();

  // A row registered under a bare `pypi` predates the per-project scheme and
  // will block every other project, since the tuple it occupies is the one they
  // would all need.
  for (const table of document.querySelectorAll(".table--publisher-list")) {
    if (/Environment name:\s*(pypi|testpypi)\s*$/m.test(table.textContent)) {
      console.warn(
        "%cOne registered publisher uses the bare environment name.%c Remove it " +
          "and re-add that project — a shared environment can only hold one " +
          "project, and it is holding the slot the others need.",
        "color:#a60;font-weight:bold",
        "color:inherit"
      );
      break;
    }
  }
})();

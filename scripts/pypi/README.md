# Filling PyPI's trusted-publisher forms

Twelve forms — six projects on pypi.org, six on test.pypi.org — differing in one
field. This types the other four and picks the environment from the hostname,
which is the field most likely to be got wrong.

**It never submits.** You read what it filled and click Add.

## Console

Open <https://pypi.org/manage/account/publishing/>, select the **GitHub** tab,
paste `fill-trusted-publisher.js` into DevTools, then:

```js
jef.next()          // fill the next project not yet done on this host
jef.fill('jef-sdk') // or pick one
jef.status()        // what is left
jef.reset()         // clear progress for this host
```

Click **Add**, and the page reloads — so re-paste and run `jef.next()` again.
Six times, then the same on test.pypi.org where the environment switches to
`testpypi` automatically.

## Bookmarklet

Avoids re-pasting after each reload. Copy the single line in `bookmarklet.txt`,
save it as a bookmark's URL, and click it on the publishing page: it fills the
next outstanding project each time.

Rebuild after editing the script:

```bash
node scripts/pypi/build-bookmarklet.mjs
```

It minifies with esbuild and parses the output before writing. Collapsing
newlines with a regex turns every `//` comment into a swallowed line, and the
resulting bookmarklet fails silently on click rather than at build time.

## Where the selectors come from

warehouse's own template, not guesswork: the form is
`#pending-github-publisher-form` and the fields are `project_name`, `owner`,
`repository`, `workflow_filename`, `environment`. Scoped to that form, so the
GitLab, Google and ActiveState tabs are untouched.

## Caveats

- Progress is kept in `localStorage` per host and only records that the script
  *typed* a project. If a submission was rejected, `jef.reset()` and redo.
- A pending publisher does not reserve the name. PyPI creates the project on
  first successful publish, so until then someone else can still take it.

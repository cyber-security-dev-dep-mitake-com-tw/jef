# Filling PyPI's trusted-publisher forms

Twelve forms — six projects on pypi.org, six on test.pypi.org. The script fills
all five fields, including the environment, which is derived from the project
and the hostname.

**Each project needs its own environment.** PyPI enforces uniqueness on
`(owner, repository, workflow, environment)`, not on the project name, so a
monorepo can register exactly one pending publisher under a shared environment
— the second is rejected with nothing but an `#errors` anchor to explain it
([warehouse#16920](https://github.com/pypi/warehouse/issues/16920)). Hence
`pypi-jef-server`, `testpypi-jef-server`, and so on, matching the per-project
environments the release workflow uses.

**It never submits, and it only does one at a time.** You read what it filled
and click **Add**.

## Use it

1. Open <https://pypi.org/manage/account/publishing/> and sign in.
2. Scroll to **Add a new pending publisher**, select the **GitHub** tab.
3. Open DevTools → Console. On the first paste Chrome and Brave refuse until you
   type `allow pasting` and press enter.
4. Paste `fill-trusted-publisher.js`, then:

```js
jef.next()          // fills the next project not already on the page
jef.status()        // what PyPI currently lists
jef.fill('jef-sdk') // or pick one
```

5. Check the fields, click **Add**. The page reloads.
6. Paste the script again and run `jef.next()`. Six times.
7. Repeat on <https://test.pypi.org/manage/account/publishing/>, where the
   environment becomes `testpypi` automatically.

**One per reload.** Calling `next()` twice without clicking Add just overwrites
the fields; nothing is registered. The script says so, and because it reads the
page rather than its own bookkeeping, it cannot claim otherwise.

## Why there is no bookmarklet

There was one, and it could never have worked. PyPI sends
`script-src 'self'` with no `unsafe-inline`, and Chromium blocks `javascript:`
URLs under such a policy — so it failed silently on click. Console execution is
exempt from CSP; bookmarklets are not.

## If you already added one with a bare `pypi` environment

Remove it and re-add. A shared environment occupies the one tuple slot every
other project would need, so nothing else can be registered while it exists.
The script warns when it spots one.

## Why progress is read from the page

The first version marked a project done the moment it typed the values, and
reported "6/6 filled" while PyPI listed exactly one — the only one that had been
submitted. Progress now comes from the **Pending publishers** table, which is
the only thing that knows what was actually registered.

## Where the selectors come from

warehouse's own template: the form is `#pending-github-publisher-form`, the
fields are `project_name`, `owner`, `repository`, `workflow_filename` and
`environment`, and the listing is `.table--publisher-list`. Filling is scoped to
that form, so the GitLab, Google and ActiveState tabs are untouched.

## One caveat that is not about this script

A pending publisher does not reserve the name. PyPI creates the project on first
successful publish, so until then anyone else can still take it — the red notice
on that page says the same thing.

# Liam Grey — portfolio site

## Viewing it on your computer

Unzip the folder first, then open `index.html` from inside the unzipped folder.
`index.html` and the `assets` folder must sit next to each other or no images load.
Opening `index.html` on its own, or previewing it inside a chat window, shows blank
image boxes — that is expected, not a broken file.

## Publishing (free, ~10 minutes)

1. Create an account at github.com. Your username becomes part of the URL, so pick
   something like `liamgrey`.
2. **New repository** → name it exactly `USERNAME.github.io` → set to **Public** → Create.
3. On the empty repo page click "uploading an existing file".
4. Drag in `index.html` and the whole `assets` folder. Commit.
5. Live at `https://USERNAME.github.io` within a minute.

Easier option: drag the whole folder onto **netlify.com/drop**.

Add the URL to your CV under your email, and to the "Website" field on LinkedIn.

## Adding the two unfinished projects

Both are already on the page as cards with cover images and an "In progress" tag.
When one is finished:

1. Change `<span class="pill wip">In progress</span>` to `<span class="pill done">Complete</span>`
2. Remove `aria-disabled="true" style="pointer-events:none;opacity:.94"` from that card's
   opening `<a>` tag so it becomes clickable.
3. Copy the whole `<article class="feature">` block, paste it below the existing one,
   give it a new `id`, and point the card's `href` at that id.

Keep the "What I built" / "What the validation caught" split. That structure is what
makes the project read as analysis rather than a screenshot.

## Editing

Everything is in `index.html`. Colours are named at the top of the `<style>` block
under `:root` — change them there and they update across the whole page.

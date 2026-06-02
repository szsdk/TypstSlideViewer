#let html-iframe-style = "width: 100%; height: 100%; border: none; transform-origin: 0 0;"
#let html-embed-mode = sys.inputs.at("html-embed-mode", default: "placeholder")

#let escape-attr(s) = s.replace("&", "&amp;").replace("\"", "&quot;").replace("<", "&lt;").replace(">", "&gt;")

#let shorten-source(source, max: 120) = {
  if source == none {
    none
  } else {
    let source = str(source).replace("\n", " ")
    if source.len() > max {
      source.slice(0, max - 3) + "..."
    } else {
      source
    }
  }
}

#let placeholder-image-key(kind, source) = {
  let key = kind + "-" + str(source)
  for char in ("\\", "/", ":", "*", "?", "\"", "<", ">", "|", "#", "%", "&", "{", "}", "$", "!", "@", "+", "=", "`", " ") {
    key = key.replace(char, "-")
  }
  key
}

#let generated-placeholder-image(kind, source) = {
  if source == none {
    none
  } else {
    let image = sys.inputs.at("html-placeholder-image-" + kind + ":" + str(source), default: none)
    if image != none {
      image
    } else {
      let dir = sys.inputs.at("html-placeholder-dir", default: ".typstslideviewer-placeholders")
      dir + "/" + placeholder-image-key(kind, source) + ".png"
    }
  }
}

#let html-content(content) = {
  let content = if content.func() == raw {
    content.text
  } else {
    content
  }
  content
}

#let xhtml(
  width: none,
  height: none,
  content,
) = {
  assert(width != none and height != none, message: "xhtml requires width and height")
  let content = html-content(content)
  let html-embed = {
    "<svg viewBox=\"0 0 "
    str(width.pt())
    " "
    str(height.pt())
    "\""
    " width=\""
    str(width.pt())
    "\" height=\""
    str(height.pt())
    "\" xmlns=\"http://www.w3.org/2000/svg\">"
    "<foreignObject width=\""
    str(width.pt())
    "\" height=\""
    str(height.pt())
    "\">"
    content
    "</foreignObject>"
    "</svg>"
  }

  image(bytes(html-embed), alt: "!typst-embed-command", width: width, height: height)
}

#let html-frame(
  width: none,
  height: none,
  content,
) = xhtml(width: width, height: height, content)

#let direct-html(
  width: none,
  height: none,
  content,
) = html-frame(width: width, height: height, content)

#let html-placeholder(
  width: none,
  height: none,
  dest: none,
  label: [Open embedded content],
  source: none,
  placeholder-image: none,
) = {
  assert(width != none and height != none, message: "html-placeholder requires width and height")
  let source = shorten-source(source)
  let body = if placeholder-image != none {
    box(
      width: width,
      height: height,
      stroke: rgb("#b8c0cc") + 0.8pt,
      clip: true,
    )[
      #if type(placeholder-image) == content {
        box(width: 100%, height: 100%)[#placeholder-image]
      } else {
        image(placeholder-image, width: 100%, height: 100%, fit: "cover")
      }
      #if dest != none {
        place(top + right, dx: -8pt, dy: 8pt)[
          #box(
            width: 24pt,
            height: 24pt,
            radius: 3pt,
            fill: white.transparentize(12%),
            stroke: rgb("#94a3b8") + 0.5pt,
            align(center + horizon, text(size: 14pt, fill: rgb("#0f172a"))[↗]),
          )
        ]
      }
      #if source != none {
        place(bottom + left, dx: 8pt, dy: -8pt)[
          #box(
            fill: white.transparentize(12%),
            stroke: rgb("#cbd5e1") + 0.5pt,
            radius: 3pt,
            inset: (x: 6pt, y: 3pt),
            text(size: 8pt, fill: rgb("#334155"), source),
          )
        ]
      }
    ]
  } else {
    box(
      width: width,
      height: height,
      stroke: rgb("#b8c0cc") + 0.8pt,
      fill: rgb("#f6f8fb"),
      inset: 18pt,
      align(center + horizon)[
        #set text(fill: rgb("#334155"))
        #stack(
          dir: ttb,
          spacing: 8pt,
          text(size: 16pt, weight: "semibold", label),
          text(size: 10pt, fill: rgb("#64748b"))[
            Interactive HTML is not available in this output.
          ],
          if source != none {
            text(size: 8pt, fill: rgb("#64748b"), source)
          },
        )
      ],
    )
  }

  if dest != none {
    link(dest, body)
  } else {
    body
  }
}

#let iframe-html(
  width: none,
  height: none,
  src: none,
  srcdoc: none,
  overlay: true,
  fallback-link: none,
  fallback-label: [Open embedded content],
  fallback-source: none,
  placeholder-image: none,
) = {
  if html-embed-mode != "iframe" {
    html-placeholder(
      width: width,
      height: height,
      dest: fallback-link,
      label: fallback-label,
      source: fallback-source,
      placeholder-image: placeholder-image,
    )
  } else {
  let overlay-attr = if overlay {
    " data-html-overlay=\"true\""
  } else {
    ""
  }
  let source-attr = if srcdoc != none {
    " srcdoc=\"" + escape-attr(srcdoc) + "\""
  } else if src != none {
    " src=\"" + escape-attr(src) + "\""
  } else {
    ""
  }
  let iframe = "<iframe" + overlay-attr + source-attr + " style=\"" + html-iframe-style + "\"></iframe>"
  html-frame(width: width, height: height, raw(iframe, lang: "html"))
  }
}

#let embed-html(
  width: none,
  height: none,
  direct: false,
  src: none,
  srcdoc: none,
  overlay: true,
  content: none,
  fallback-link: none,
  fallback-label: [Open embedded content],
  placeholder-image: none,
) = {
  if direct {
    direct-html(width: width, height: height, content)
  } else {
    let fallback-link = if fallback-link != none { fallback-link } else { src }
    let placeholder-image = if placeholder-image != none {
      placeholder-image
    } else if src != none {
      generated-placeholder-image("src", src)
    } else if srcdoc != none {
      generated-placeholder-image("srcdoc", srcdoc)
    } else {
      none
    }
    let fallback-source = if src != none {
      "src: " + str(src)
    } else if srcdoc != none {
      "srcdoc: " + srcdoc
    } else {
      none
    }
    iframe-html(
      width: width,
      height: height,
      src: src,
      srcdoc: srcdoc,
      overlay: overlay,
      fallback-link: fallback-link,
      fallback-label: fallback-label,
      fallback-source: fallback-source,
      placeholder-image: placeholder-image,
    )
  }
}

#let embed-html-file(
  width: none,
  height: none,
  path: none,
  overlay: true,
  fallback-link: none,
  fallback-label: [Open embedded content],
  placeholder-image: none,
) = {
  assert(path != none, message: "embed-html-file requires path")
  let fallback-link = if fallback-link != none { fallback-link } else { path }
  let placeholder-image = if placeholder-image != none {
    placeholder-image
  } else {
    generated-placeholder-image("path", path)
  }
  iframe-html(
    width: width,
    height: height,
    srcdoc: read(path),
    overlay: overlay,
    fallback-link: fallback-link,
    fallback-label: fallback-label,
    fallback-source: "path: " + str(path),
    placeholder-image: placeholder-image,
  )
}

#let embed-video(
  width: none,
  height: none,
  src: none,
  mime: "video/mp4",
  controls: true,
  autoplay: false,
  loop: false,
  muted: false,
  fallback-link: none,
  fallback-label: [Open video],
  placeholder-image: none,
) = {
  assert(src != none, message: "embed-video requires src")
  if html-embed-mode != "iframe" {
    let fallback-link = if fallback-link != none { fallback-link } else { src }
    let placeholder-image = if placeholder-image != none {
      placeholder-image
    } else {
      generated-placeholder-image("video", src)
    }
    html-placeholder(
      width: width,
      height: height,
      dest: fallback-link,
      label: fallback-label,
      source: "src: " + str(src),
      placeholder-image: placeholder-image,
    )
  } else {
  let controls-attr = if controls { " controls=\"\"" } else { "" }
  let autoplay-attr = if autoplay { " autoplay=\"\"" } else { "" }
  let loop-attr = if loop { " loop=\"\"" } else { "" }
  let muted-attr = if muted { " muted=\"\"" } else { "" }
  let video = {
    "<video xmlns=\"http://www.w3.org/1999/xhtml\" width=\"100%\" height=\"100%\""
    controls-attr
    autoplay-attr
    loop-attr
    muted-attr
    ">"
    "<source src=\""
    escape-attr(src)
    "\" type=\""
    escape-attr(mime)
    "\" />"
    "</video>"
  }
  direct-html(width: width, height: height, raw(video, lang: "html"))
  }
}

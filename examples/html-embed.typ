#let html-iframe-style = "width: 100%; height: 100%; border: none; transform-origin: 0 0;"

#let escape-attr(s) = s.replace("&", "&amp;").replace("\"", "&quot;").replace("<", "&lt;").replace(">", "&gt;")

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

#let iframe-html(
  width: none,
  height: none,
  src: none,
  srcdoc: none,
  overlay: true,
) = {
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

#let embed-html(
  width: none,
  height: none,
  direct: false,
  src: none,
  srcdoc: none,
  overlay: true,
  content: none,
) = {
  if direct {
    direct-html(width: width, height: height, content)
  } else {
    iframe-html(width: width, height: height, src: src, srcdoc: srcdoc, overlay: overlay)
  }
}

#let embed-html-file(
  width: none,
  height: none,
  path: none,
  overlay: true,
) = {
  assert(path != none, message: "embed-html-file requires path")
  embed-html(width: width, height: height, srcdoc: read(path), overlay: overlay)
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
) = {
  assert(src != none, message: "embed-video requires src")
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

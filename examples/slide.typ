#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge, shapes
#import "@preview/touying:0.5.5": *
#import "@preview/cetz:0.3.1"
#import themes.metropolis: *

#set cite(style: "chicago-notes")
// #set text(font: "Arial")
#set text(font: "Fira Sans")

// #set text(font: "Lete Sans Math")
#show math.equation: set text(font: "Lete Sans Math", weight: "regular")
#show: metropolis-theme.with(
  aspect-ratio: "16-9",
  footer: self => self.info.institution,
  config-info(
    title: [Typst Slide Viewer],
    subtitle: [],
    author: [szsdk],
  ),
  // handout: true
  // config-common(show-notes-on-second-screen: right)
)

#let xhtml(outer-width: 1024pt, outer-height: 768pt, inner-width: none, inner-height: none, content) = {
  let t = content.func()
  let content = if content.func() == raw {
    content.text
  } else {
    content
  }

  let inner-width = if inner-width == none {
    outer-width
  } else {
    inner-width
  }

  let inner-height = if inner-height == none {
    outer-height
  } else {
    inner-height
  }

  let html-embed = {
    "<svg viewBox=\"0 0 "
    str(inner-width.pt())
    " "
    str(inner-height.pt())
    "\""
    " width=\""
    str(outer-width.pt())
    "\" height=\""
    str(outer-height.pt())
    "\" xmlns=\"http://www.w3.org/2000/svg\">"
    "<foreignObject width=\""
    str(inner-width.pt())
    "\" height=\""
    str(inner-height.pt())
    "\">"
    content
    "</foreignObject>"
    "</svg>"
  }

  image.decode(html-embed, alt: "!typst-embed-command", width: outer-width, height:outer-height)
}

#title-slide()

== Add foreignObject 
#xhtml(outer-width: 400pt, outer-height: 400pt, inner-height: 400pt, inner-width: 400pt, ```html
  <iframe src="https://typst.app" width="200%" height="160%" style="border: none; transform: scale(0.6); transform-origin: 0 0;"></iframe>
  ```)

== Add video
#xhtml(outer-width: 400pt, outer-height: 400pt, inner-height: 400pt, inner-width: 400pt, ```html
  <video xmlns="http://www.w3.org/1999/xhtml" width="400px" height="400px" controls="" >
    <source src="./sample-5s.mp4" type="video/mp4" />
  </video>
  ```)

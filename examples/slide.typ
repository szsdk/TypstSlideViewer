#import "@preview/fletcher:0.5.3" as fletcher: diagram, node, edge, shapes
#import "@preview/touying:0.6.2": *
#import "@preview/cetz:0.3.1"
#import themes.metropolis: *
#import "html-embed.typ": embed-html, embed-html-file, embed-video

#set cite(style: "chicago-notes")
// #set text(font: "Arial")
// #set text(font: "Fira Sans")
#set text(font: ("Fira Sans", "Noto Sans CJK SC"))

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

#title-slide()

== read foreignObject 

#align(center,
  embed-html-file(width: 680pt, height: 382.5pt, path: "three.html")
)

== Add foreignObject 
#slide(
  [
    #align(center,embed-html(width: 380pt, height: 182.5pt, src: "./three.html"))
    #pause
    hello
  ]
)

== 中文测试嘿嘿 #emoji.ambulance
#slide(config: config-page(background: image("r.jpg", width:100%)))[
  #box(fill:white.transparentize(60%))[
  $
  and.big_i^infinity bold(x)_i = integral.cont bold(f)(v) dot dif bold(s)
  $
]
  #align(left,
  embed-video(width: 400pt, height: 300pt, src: "./sample-5s.mp4")
)
)
]

== Image
#slide(config: config-page(background: image("r.jpg", width:100%)))[
#image("r.jpg", width: 60%)
]

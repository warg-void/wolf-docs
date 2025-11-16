import { defaultTheme } from '@vuepress/theme-default'
import { defineUserConfig } from 'vuepress'
import { viteBundler } from '@vuepress/bundler-vite'
import { markdownMathPlugin } from '@vuepress/plugin-markdown-math'

const sidebar = [
  {
    text: 'Installation',
    
    children: [
      '/install/readme.md',
    ],
  },
  {
    text: 'Examples',
    // collapsible: true,
    // collapsed: true,
    children: [
      '/examples/mnist.md',
    ],
  },
]

export default defineUserConfig({
  lang: 'en-US',

  title: 'Wolf',
  description: 'Simple and Native C++ ML Library',
  base: '/wolf-docs/',
  theme: defaultTheme({
    logo: '/images/swhappy.png',
    colorMode: 'dark',
    colorModeSwitch: 'true',
    repo:'warg-void/wolf',
    navbar: [
    { text: 'Home', link: '/' },
    { text: 'Get Started', link: '/install/' },
    { text: 'Examples', link:'examples/mnist'}
    ],
    sidebar,
  }),


  bundler: viteBundler(),
  plugins: [
    markdownMathPlugin({

    }),
  ],
})

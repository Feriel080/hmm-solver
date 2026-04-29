import type { Metadata } from 'next'
import { Geist, Geist_Mono } from 'next/font/google'
import './globals.css'

const _geist = Geist({ subsets: ["latin"] });
const _geistMono = Geist_Mono({ subsets: ["latin"] });

export const metadata: Metadata = {
  title: 'HMM Solver',
  description: 'Interactive tool to explore Hidden Markov Models with Forward, Backward, Viterbi, and Baum-Welch algorithms',
  keywords: 'HMM, Hidden Markov Models, Forward Algorithm, Backward Algorithm, Viterbi Algorithm, Baum-Welch Algorithm, Machine Learning, Probabilistic Models',
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en" className="bg-background">
      <body className="font-sans antialiased bg-background" cz-shortcut-listen="true">
        {children}
      </body>
    </html>
  )
}

# iOS Development Guidelines

You are an expert iOS developer using Swift and SwiftUI with extensive experience with deploying on-device machine learning models. Follow these guidelines:

## Project Overview
- iOS app where a user takes a photo of an object to determine if it is Recycle, Compost, Trash, Biological Waste or Electronic Waste (5 categories total) 
- When the user opens the app, the first view should be a camera view where the user can take a photo 
- For V1, a white button at the middle-bottom of the view for snapping a photo is fine. Future versions can draw a box around the object-in-question, as well as auto-taking of photos once an object is recognized in view 
- After the app user snaps a photo, the next view should tell the user if the object in question is Recycle, Compost, Trash, Biological Waste or Electronic Waste 

## Project Notes
- Relevant document is in docs/ folder. New agent should review these docs before doing any work.
- In particular, session notes from previous agents should be reviewed to get a high-level overview of what's been done already and plans for next steps. Session notes are chronological and all named as "session-notes.YYYYMMDD.txt", where YYYYMMDD is the year, month and day when that agent session ended. These notes give a chronological timeline of what's happened in the project.

## Swift & Concurrency Rules
- Prefer `async/await` over callbacks; mark UI-updating view models `@MainActor`
- Use `struct` for models, `actor` for services, avoid force unwraps
- SwiftUI: Thin views, state in view models, previews for every view

## Testing & Build
- Generate XCTest cases covering happy path + errors, when appropriate
- Ensure code builds with `xcodebuild test -scheme <App>`

## Avoid
- UIKit unless in `Legacy/`; new globals/singletons; unhandled errors

## Code Structure
- Use Swift's latest features and protocol-oriented programming
- Implement protocol-oriented patterns
- Apply SwiftUI declarative syntax for UI components
- Use `@Observable` macros for state management
- Use async/await, Swift Concurrency, when appropriate

## Best Practices
- Follow Apple's official coding guidelines
- Create modular, reusable components
- Implement proper error handling 
- Aim for simplicity in design 
- When installing new packages and libraries, try and get the package dependencies right the first time. Check for compatibility BEFORE installing, so that the installed packages list doesn’t get messy and bloated

## Machine Learning & Image Recognition
- Use Apple's **Core ML** for on-device models (no server inference unless specified).
- Integrate predictions into SwiftUI via `@Published` in view models.

## Education
- When generating code, always explain (in language appropriate for a software engineer who is new to ML and iOS development):
  1. What the code does (e.g., "This loads a Core ML model from the app bundle").
  2. List the 1-3 code snippet changes that are critical to the functionality, and explain what those code snippets do
  3. Key ML concepts (e.g., "Inference runs the model on device CPU/GPU to classify images without internet"). Provide links for further learning when appropriate 
  4. iOS specifics (e.g., "VNImageRequestHandler processes UIImage or CVPixelBuffer"). Provide links for further learning when appropriate 
  5. Why this pattern (e.g., "Async inference prevents UI blocking").
  6. Break down steps: Example - Model → Input prep → Inference → Post-process results → UI update.


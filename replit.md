# HANDZ - The Boxing Game

## Overview
HANDZ is a full-stack web application implementing a top-down orthographic boxing game. It features a canvas-based fighting system with 8-directional movement, stamina-driven combat, diverse fighter archetypes, and a career mode that includes persistent fighter progression. The game utilizes HTML5 Canvas for real-time rendering and a PostgreSQL database for managing fighter data and fight outcomes. The project aims to deliver a comprehensive boxing simulation experience in a web browser.

## User Preferences
Preferred communication style: Simple, everyday language.

## System Architecture

### Frontend
The frontend is built with React 18, TypeScript, Wouter for routing, and React Query for server state management. UI components are styled with shadcn/ui (New York style) and Tailwind CSS, supporting light/dark modes. The game engine is custom-built on HTML5 Canvas, featuring an advanced AI system with state machines, tactical phases, and personality, ported from a C# Unity implementation. Key game mechanics include a 2.5D perspective, 8-directional fighter movement, a rhythm-based combat system, feint-whiff punishment, a 5-phase punch system, and detailed defense mechanics. Fighter customization includes gear colors. A comprehensive knockdown system with ref mercy stoppages and corner towel stoppages is implemented. Quick Fight mode offers adjustable difficulty and round settings, plus a "Select Opponent" roster picker to choose specific fighters from the 204-fighter roster. A two-stage Tutorial Mode accessible from the main menu teaches core mechanics (movement, punches, blocking, ducking) in Stage 1 and advanced techniques (auto guard, guard toggling, weaving, rhythm control) in Stage 2, using a Journeyman level-10 AI opponent that starts idle and activates after punch tutorials. Career Mode features an open-ended career with ELO-based rankings, a 204-fighter roster, weekly simulations, XP-based progression with stat point allocation, and a prep weeks system. The interactive Gym is the career home screen: loading a save, finishing a fight, training, or simulating a week all land in the gym. It shows a fighter header (name/record/rank), a clickable Force chip that opens a Force→SP convert dialog (1,000 Force = 1 SP), an office desk that opens the Fight Planner (the former hub screen, now stripped of training/convert controls), two wide trophy cases flanking the exit door (left → Career Stats, right → Skill Refinement, locked until rank 650) that fill with trophies (1 per 3 wins, 30/case) and medals (1 per 20 refinement points, 40/case) persisted per save in localStorage (`handz_trophies_<fighterId>`), the player's own fighter idling by a wooden bench (click → Edit Colors), and a weekly +XP/+SP bonus tag over the matching equipment (hidden during fight week or when the 2-sessions-per-week training limit is reached, which also locks the Spar/Train/Bag Work popup buttons). The gym exit door returns to the main menu. All sparring modes (every difficulty, Nightmare, and Doghouse) take place in the gym scene: the gym equipment is drawn around the ring during sparring fights and follows the live fight camera (yaw tracking, auto-zoom, focus follow), while each mode's UI and HUD are unchanged. An Input Recording system captures detailed per-frame fight data for AI training analysis, exportable as text files. The game also includes a procedural audio engine built on the Web Audio API for all game sounds and a pre-punch telegraph animation system that scales with fighter level. A visual neural network editor allows admin-level control over AI behavior parameters per difficulty. Per-fighter neural networks (no PIN required) allow individual AI behavior customization. A "Reset All Networks" button with confirmation clears all global, default, and per-fighter neural overrides. A player playstyle visualization in the Career Stats section displays a read-only radar chart computed from fight/sparring performance data, using the same 22 neural network parameters — it updates after each fight or sparring session and is stored in localStorage (`handz_player_playstyle`) with exponential moving average blending (60% old / 40% new). An Invisible Adaptive AI Behavior Engine runs under the hood: AI opponents gradually learn the player's effective patterns during a fight and subtly adjust tactics. The system tracks micro-patterns (jab-step, duck-counter, pivot-punch, backstep-counter, block-counter, dodge-counter) and macro-patterns (exchanges with combo sequences, ring position context) with outcome gating — only patterns where the AI takes more damage or stamina loss are forwarded for adaptation. A 32-slot timing base on AiBrainState stores nudgeable AI response preferences (commit chance, patience, feint bias, body targeting, uppercut-on-duck, retreat tracking, guard drop exploit, engage cycle timing, defense discipline, etc.). Adaptation is probabilistic per difficulty (Journeyman 2%, Contender 8%, Elite 18%, Champion 30%), reviewed every 10 seconds mid-round, with risk management scaling. Confidence decays 0.85x per round boundary and round 1 adaptations are weak. An "ADAPTIVE AI MODE" toggle in MainMenu settings (localStorage key `handz_adaptive_ai`, default true) controls the entire system.

### Backend
The backend is an Express 5 server using Node.js and TypeScript, exposing a RESTful JSON API for fighter management and fight result storage.

### Shared Code
A `shared/` directory contains Drizzle ORM schema definitions and Zod validation schemas for `fighters` and `fight_results` tables, ensuring type safety and consistency.

### Database
PostgreSQL serves as the primary data store, managed by Drizzle ORM. The schema includes `fighters` (with detailed stats, progression, and career data) and `fight_results` tables.

### Build System
Vite is used for frontend development and building, while esbuild handles the backend. Path aliases simplify module imports.

## External Dependencies

### Database
- **PostgreSQL**: Primary database.
- **pg**: Node.js client for PostgreSQL.

### Key NPM Packages
- **drizzle-orm** & **drizzle-kit**: ORM for database interaction and schema management.
- **@tanstack/react-query**: For server state management in React.
- **wouter**: Lightweight client-side router.
- **zod**: Schema validation.
- **shadcn/ui ecosystem**: UI component library and styling utilities.
- **express**: Backend web framework.
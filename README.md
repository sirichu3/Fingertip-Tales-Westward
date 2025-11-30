# Fingertip-Tales-Westward
A Fun Shadow Puppet Experience Project

## Overview
- Category: Interactive experience inspired by galgame
- Art Style: Chinese traditional shadow puppetry
- Concept: This project reimagines Journey to the West through multi-perspective storytelling and tactile performance. It adapts the classic "Sun Wukong Borrows the Banana Fan Three Times" into an interactive experience with shadow-puppet aesthetics.
- Narrative: Uses three-line storytelling, shifting perspectives among monsters, the heavenly court, and other roles, allowing players to reinterpret the classic through fresh viewpoints while appreciating the artistry of shadow puppetry.

## Core Gameplay
- Players use camera-based hand gesture recognition to select and directly control shadow-puppet characters, performing set pieces and poses to progress.
- Combines traditional visual aesthetics with motion-driven interaction to deliver an immersive cultural and entertainment experience.

## Project Structure
- Developer folders: `bone/`, `collision_volume/`, `pose/`
  - Visual tools to bind puppet bones, attach collision volumes, and define puppet poses.
- Assets: `Audio/`, `video/`, `vision_resources/`
- Entrypoints: `open/`, `main_menu/`, `handpose2/`
  - Contains direct scene launchers and demos.

## How to Run
- Full flow: start from `open/hand_gate.py` and follow the on-screen guidance through the experience.
- Story mode: run `handpose2/story1.py` to jump directly into the story.
- Puppet-only demos: run `handpose2/demo.py` or `handpose2/demo copy.py` to experience shadow-puppet control.

## Notes
- Gesture recognition may be unreliable in dark environments; play in a well-lit space.
- Current visuals and interactions are a work-in-progress and do not represent final quality.

---

# 皮影互动体验 — 指尖灯影集·西游

## 概述
- 作品分类：借鉴 galgame 形式的互动体验项目
- 美术风格：中国传统皮影戏
- 作品介绍：本项目以《西游记》经典片段为原型，融合皮影戏的独特美术风格，打造别具一格的中式文化体验。《孙悟空三借芭蕉扇》通过改变叙述视角，采用三线叙事，让玩家从妖怪、天庭等不同角色的视角重新解读经典，感受传统皮影戏的艺术魅力。

## 核心玩法
- 玩家通过摄像头识别手势，选择并亲手操控皮影人物完成剧目表演与姿态匹配。
- 将传统视觉美学与体感互动结合，提供沉浸式的文化与娱乐体验。

## 项目结构
- 开发者文件夹：`bone/`、`collision_volume/`、`pose/`
  - 用于视觉化地为人偶绑骨、给物体绑定碰撞体积、定义人偶 pose。
- 资源文件：`Audio/`、`video/`、`vision_resources/`
- 入口目录：`open/`、`main_menu/`、`handpose2/`
  - 内含各界面的直接入口与示例。

## 使用方法
- 完整流程：从 `open/hand_gate.py` 进入，并按指引逐步体验。
- 直接看故事：运行 `handpose2/story1.py`。
- 只玩皮影：体验 `handpose2/demo.py` 与 `handpose2/demo copy.py`。

## 注释
- 在黑暗环境下手势识别可能出现问题，建议在明亮环境下体验。
- 当前效果不代表最终质量，后续会持续优化。

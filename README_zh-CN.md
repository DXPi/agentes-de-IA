# Auto-GPT: 一个实验性的用于自动化GPT-4
[![Unit Tests](https://img.shields.io/github/actions/workflow/status/Significant-Gravitas/Auto-GPT/ci.yml?label=unit%20tests)](https://github.com/Significant-Gravitas/Auto-GPT/actions/workflows/ci.yml)
[![Discord Follow](https://dcbadge.vercel.app/api/server/autogpt?style=flat)](https://discord.gg/autogpt)
[![GitHub Repo stars](https://img.shields.io/github/stars/Significant-Gravitas/auto-gpt?style=social)](https://github.com/Significant-Gravitas/Auto-GPT/stargazers)
[![Twitter Follow](https://img.shields.io/twitter/follow/siggravitas?style=social)](https://twitter.com/SigGravitas)

[English](./README.md) 简体中文

## 💡 你可以在 [常见问题](https://github.com/Significant-Gravitas/Auto-GPT/discussions/categories/q-a) 还有 [Discord服务器 💬](https://discord.gg/autogpt) 上获取帮助

<hr/>

###  🔴 🔴  警告: 使用*稳定* 的发行版而不是`master`分支 🔴 🔴 

**在这里下载最新的稳定发行版: https://github.com/Significant-Gravitas/Auto-GPT/releases/latest.**
`master`分支可能经常处于 **不可用** 状态，请不要使用`master`分支

<hr/>


Auto-GPT 是一个实验性开源软件，为你展示 GPT-4 模型的更多可能。 该程序由 GPT-4 驱动，使用一种特殊的“语言”与 GPT-4 进行沟通，将 LLM 的“思想”链接在一起，以自动实现您设定的目标。 作为首批可以让 GPT-4 自动完成任务的项目之一，Auto-GPT 让 AI 上升到了一个新的捣鼓。

<h2 align="center"> 一个示例（2023 年 4 月 16 日发布） </h2>

https://user-images.githubusercontent.com/70048414/232352935-55c6bf7c-3958-406e-8610-0913475a0b05.mp4

感谢 <a href=https://twitter.com/BlakeWerlinger>Blake Werlinger</a> 创作的示例

<h2 align="center"> 💖 资助我们以帮助 Auto GPT 的发展 💖</h2>
<p align="center">
如果你愿意放弃一杯奶茶, 你就可以帮助 Auto GPT 的发展，让我们有动力让 AI 自动帮助你完成更多任务。
您的支持是对 <a href="https://github.com/Significant-Gravitas/Auto-GPT/graphs/contributors">开发者们</a> 的肯定
这是我们的 <a href="https://github.com/sponsors/Torantulino">赞助商们</a>.
如果你愿意赞助请点击 <a href="https://github.com/sponsors/Torantulino">这里</a>，我们会让您个人、公司的头像出现在这里。
</p>



<p align="center">
<div align="center" class="logo-container">
<a href="https://www.zilliz.com/">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234158272-7917382e-ff80-469e-8d8c-94f4477b8b5a.png">
  <img src="https://user-images.githubusercontent.com/22963551/234158222-30e2d7a7-f0a9-433d-a305-e3aa0b194444.png" height="40px" alt="Zilliz" />
</picture>
</a>
<a href="https://roost.ai">
<img src="https://user-images.githubusercontent.com/22963551/234180283-b58cb03c-c95a-4196-93c1-28b52a388e9d.png" height="40px" alt="Roost.AI" />
</a>
<a href="https://nuclei.ai/">

<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234153428-24a6f31d-c0c6-4c9b-b3f4-9110148f67b4.png">
  <img src="https://user-images.githubusercontent.com/22963551/234181283-691c5d71-ca94-4646-a1cf-6e818bd86faa.png" height="40px" alt="NucleiAI" />
</picture>
</a>

<a href="https://www.algohash.org/">

<picture>
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234180375-1365891c-0ba6-4d49-94c3-847c85fe03b0.png" >
  <img src="https://user-images.githubusercontent.com/22963551/234180359-143e4a7a-4a71-4830-99c8-9b165cde995f.png" height="40px" alt="Algohash" />
</picture>
</a>

<a href="https://www.typingmind.com/?utm_source=autogpt">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/233202971-61e77209-58a0-47d9-9f7e-dd081111437b.png">
  <img src="https://user-images.githubusercontent.com/22963551/234157731-f908b5db-8fe7-4036-89b6-7b2a21f87e3a.png" height="40px" alt="TypingMind" />
</picture>
</a>

<a href="https://github.com/weaviate/weaviate">
<picture height="40px">
  <source media="(prefers-color-scheme: light)" srcset="https://user-images.githubusercontent.com/22963551/234181699-3d7f6ea8-5a7f-4e98-b812-37be1081be4b.png">
  <img src="https://user-images.githubusercontent.com/22963551/234181695-fc895159-b921-4895-9a13-65e6eff5b0e7.png" height="40px" alt="TypingMind" />
</picture>
</a>

</div>
</br>



<p align="center"><a href="https://github.com/robinicus"><img src="https://avatars.githubusercontent.com/robinicus?v=4" width="50px" alt="robinicus" /></a>&nbsp;&nbsp;<a href="https://github.com/0xmatchmaker"><img src="https://avatars.githubusercontent.com/0xmatchmaker?v=4" width="50px" alt="0xmatchmaker" /></a>&nbsp;&nbsp;<a href="https://github.com/jazgarewal"><img src="https://avatars.githubusercontent.com/jazgarewal?v=4" width="50px" alt="jazgarewal" /></a>&nbsp;&nbsp;<a href="https://github.com/MayurVirkar"><img src="https://avatars.githubusercontent.com/MayurVirkar?v=4" width="50px" alt="MayurVirkar" /></a>&nbsp;&nbsp;<a href="https://github.com/avy-ai"><img src="https://avatars.githubusercontent.com/avy-ai?v=4" width="50px" alt="avy-ai" /></a>&nbsp;&nbsp;<a href="https://github.com/TheStoneMX"><img src="https://avatars.githubusercontent.com/TheStoneMX?v=4" width="50px" alt="TheStoneMX" /></a>&nbsp;&nbsp;<a href="https://github.com/goldenrecursion"><img src="https://avatars.githubusercontent.com/goldenrecursion?v=4" width="50px" alt="goldenrecursion" /></a>&nbsp;&nbsp;<a href="https://github.com/MatthewAgs"><img src="https://avatars.githubusercontent.com/MatthewAgs?v=4" width="50px" alt="MatthewAgs" /></a>&nbsp;&nbsp;<a href="https://github.com/eelbaz"><img src="https://avatars.githubusercontent.com/eelbaz?v=4" width="50px" alt="eelbaz" /></a>&nbsp;&nbsp;<a href="https://github.com/rapidstartup"><img src="https://avatars.githubusercontent.com/rapidstartup?v=4" width="50px" alt="rapidstartup" /></a>&nbsp;&nbsp;<a href="https://github.com/gklab"><img src="https://avatars.githubusercontent.com/gklab?v=4" width="50px" alt="gklab" /></a>&nbsp;&nbsp;<a href="https://github.com/VoiceBeer"><img src="https://avatars.githubusercontent.com/VoiceBeer?v=4" width="50px" alt="VoiceBeer" /></a>&nbsp;&nbsp;<a href="https://github.com/DailyBotHQ"><img src="https://avatars.githubusercontent.com/DailyBotHQ?v=4" width="50px" alt="DailyBotHQ" /></a>&nbsp;&nbsp;<a href="https://github.com/lucas-chu"><img src="https://avatars.githubusercontent.com/lucas-chu?v=4" width="50px" alt="lucas-chu" /></a>&nbsp;&nbsp;<a href="https://github.com/knifour"><img src="https://avatars.githubusercontent.com/knifour?v=4" width="50px" alt="knifour" /></a>&nbsp;&nbsp;<a href="https://github.com/refinery1"><img src="https://avatars.githubusercontent.com/refinery1?v=4" width="50px" alt="refinery1" /></a>&nbsp;&nbsp;<a href="https://github.com/st617"><img src="https://avatars.githubusercontent.com/st617?v=4" width="50px" alt="st617" /></a>&nbsp;&nbsp;<a href="https://github.com/neodenit"><img src="https://avatars.githubusercontent.com/neodenit?v=4" width="50px" alt="neodenit" /></a>&nbsp;&nbsp;<a href="https://github.com/CrazySwami"><img src="https://avatars.githubusercontent.com/CrazySwami?v=4" width="50px" alt="CrazySwami" /></a>&nbsp;&nbsp;<a href="https://github.com/Heitechsoft"><img src="https://avatars.githubusercontent.com/Heitechsoft?v=4" width="50px" alt="Heitechsoft" /></a>&nbsp;&nbsp;<a href="https://github.com/RealChrisSean"><img src="https://avatars.githubusercontent.com/RealChrisSean?v=4" width="50px" alt="RealChrisSean" /></a>&nbsp;&nbsp;<a href="https://github.com/abhinav-pandey29"><img src="https://avatars.githubusercontent.com/abhinav-pandey29?v=4" width="50px" alt="abhinav-pandey29" /></a>&nbsp;&nbsp;<a href="https://github.com/Explorergt92"><img src="https://avatars.githubusercontent.com/Explorergt92?v=4" width="50px" alt="Explorergt92" /></a>&nbsp;&nbsp;<a href="https://github.com/SparkplanAI"><img src="https://avatars.githubusercontent.com/SparkplanAI?v=4" width="50px" alt="SparkplanAI" /></a>&nbsp;&nbsp;<a href="https://github.com/crizzler"><img src="https://avatars.githubusercontent.com/crizzler?v=4" width="50px" alt="crizzler" /></a>&nbsp;&nbsp;<a href="https://github.com/kreativai"><img src="https://avatars.githubusercontent.com/kreativai?v=4" width="50px" alt="kreativai" /></a>&nbsp;&nbsp;<a href="https://github.com/omphos"><img src="https://avatars.githubusercontent.com/omphos?v=4" width="50px" alt="omphos" /></a>&nbsp;&nbsp;<a href="https://github.com/Jahmazon"><img src="https://avatars.githubusercontent.com/Jahmazon?v=4" width="50px" alt="Jahmazon" /></a>&nbsp;&nbsp;<a href="https://github.com/tjarmain"><img src="https://avatars.githubusercontent.com/tjarmain?v=4" width="50px" alt="tjarmain" /></a>&nbsp;&nbsp;<a href="https://github.com/ddtarazona"><img src="https://avatars.githubusercontent.com/ddtarazona?v=4" width="50px" alt="ddtarazona" /></a>&nbsp;&nbsp;<a href="https://github.com/saten-private"><img src="https://avatars.githubusercontent.com/saten-private?v=4" width="50px" alt="saten-private" /></a>&nbsp;&nbsp;<a href="https://github.com/anvarazizov"><img src="https://avatars.githubusercontent.com/anvarazizov?v=4" width="50px" alt="anvarazizov" /></a>&nbsp;&nbsp;<a href="https://github.com/lazzacapital"><img src="https://avatars.githubusercontent.com/lazzacapital?v=4" width="50px" alt="lazzacapital" /></a>&nbsp;&nbsp;<a href="https://github.com/m"><img src="https://avatars.githubusercontent.com/m?v=4" width="50px" alt="m" /></a>&nbsp;&nbsp;<a href="https://github.com/Pythagora-io"><img src="https://avatars.githubusercontent.com/Pythagora-io?v=4" width="50px" alt="Pythagora-io" /></a>&nbsp;&nbsp;<a href="https://github.com/Web3Capital"><img src="https://avatars.githubusercontent.com/Web3Capital?v=4" width="50px" alt="Web3Capital" /></a>&nbsp;&nbsp;<a href="https://github.com/toverly1"><img src="https://avatars.githubusercontent.com/toverly1?v=4" width="50px" alt="toverly1" /></a>&nbsp;&nbsp;<a href="https://github.com/digisomni"><img src="https://avatars.githubusercontent.com/digisomni?v=4" width="50px" alt="digisomni" /></a>&nbsp;&nbsp;<a href="https://github.com/concreit"><img src="https://avatars.githubusercontent.com/concreit?v=4" width="50px" alt="concreit" /></a>&nbsp;&nbsp;<a href="https://github.com/LeeRobidas"><img src="https://avatars.githubusercontent.com/LeeRobidas?v=4" width="50px" alt="LeeRobidas" /></a>&nbsp;&nbsp;<a href="https://github.com/Josecodesalot"><img src="https://avatars.githubusercontent.com/Josecodesalot?v=4" width="50px" alt="Josecodesalot" /></a>&nbsp;&nbsp;<a href="https://github.com/dexterityx"><img src="https://avatars.githubusercontent.com/dexterityx?v=4" width="50px" alt="dexterityx" /></a>&nbsp;&nbsp;<a href="https://github.com/rickscode"><img src="https://avatars.githubusercontent.com/rickscode?v=4" width="50px" alt="rickscode" /></a>&nbsp;&nbsp;<a href="https://github.com/Brodie0"><img src="https://avatars.githubusercontent.com/Brodie0?v=4" width="50px" alt="Brodie0" /></a>&nbsp;&nbsp;<a href="https://github.com/FSTatSBS"><img src="https://avatars.githubusercontent.com/FSTatSBS?v=4" width="50px" alt="FSTatSBS" /></a>&nbsp;&nbsp;<a href="https://github.com/nocodeclarity"><img src="https://avatars.githubusercontent.com/nocodeclarity?v=4" width="50px" alt="nocodeclarity" /></a>&nbsp;&nbsp;<a href="https://github.com/jsolejr"><img src="https://avatars.githubusercontent.com/jsolejr?v=4" width="50px" alt="jsolejr" /></a>&nbsp;&nbsp;<a href="https://github.com/amr-elsehemy"><img src="https://avatars.githubusercontent.com/amr-elsehemy?v=4" width="50px" alt="amr-elsehemy" /></a>&nbsp;&nbsp;<a href="https://github.com/RawBanana"><img src="https://avatars.githubusercontent.com/RawBanana?v=4" width="50px" alt="RawBanana" /></a>&nbsp;&nbsp;<a href="https://github.com/horazius"><img src="https://avatars.githubusercontent.com/horazius?v=4" width="50px" alt="horazius" /></a>&nbsp;&nbsp;<a href="https://github.com/SwftCoins"><img src="https://avatars.githubusercontent.com/SwftCoins?v=4" width="50px" alt="SwftCoins" /></a>&nbsp;&nbsp;<a href="https://github.com/tob-le-rone"><img src="https://avatars.githubusercontent.com/tob-le-rone?v=4" width="50px" alt="tob-le-rone" /></a>&nbsp;&nbsp;<a href="https://github.com/RThaweewat"><img src="https://avatars.githubusercontent.com/RThaweewat?v=4" width="50px" alt="RThaweewat" /></a>&nbsp;&nbsp;<a href="https://github.com/jun784"><img src="https://avatars.githubusercontent.com/jun784?v=4" width="50px" alt="jun784" /></a>&nbsp;&nbsp;<a href="https://github.com/joaomdmoura"><img src="https://avatars.githubusercontent.com/joaomdmoura?v=4" width="50px" alt="joaomdmoura" /></a>&nbsp;&nbsp;<a href="https://github.com/rejunity"><img src="https://avatars.githubusercontent.com/rejunity?v=4" width="50px" alt="rejunity" /></a>&nbsp;&nbsp;<a href="https://github.com/mathewhawkins"><img src="https://avatars.githubusercontent.com/mathewhawkins?v=4" width="50px" alt="mathewhawkins" /></a>&nbsp;&nbsp;<a href="https://github.com/caitlynmeeks"><img src="https://avatars.githubusercontent.com/caitlynmeeks?v=4" width="50px" alt="caitlynmeeks" /></a>&nbsp;&nbsp;<a href="https://github.com/jd3655"><img src="https://avatars.githubusercontent.com/jd3655?v=4" width="50px" alt="jd3655" /></a>&nbsp;&nbsp;<a href="https://github.com/Odin519Tomas"><img src="https://avatars.githubusercontent.com/Odin519Tomas?v=4" width="50px" alt="Odin519Tomas" /></a>&nbsp;&nbsp;<a href="https://github.com/DataMetis"><img src="https://avatars.githubusercontent.com/DataMetis?v=4" width="50px" alt="DataMetis" /></a>&nbsp;&nbsp;<a href="https://github.com/webbcolton"><img src="https://avatars.githubusercontent.com/webbcolton?v=4" width="50px" alt="webbcolton" /></a>&nbsp;&nbsp;<a href="https://github.com/rocks6"><img src="https://avatars.githubusercontent.com/rocks6?v=4" width="50px" alt="rocks6" /></a>&nbsp;&nbsp;<a href="https://github.com/cxs"><img src="https://avatars.githubusercontent.com/cxs?v=4" width="50px" alt="cxs" /></a>&nbsp;&nbsp;<a href="https://github.com/fruition"><img src="https://avatars.githubusercontent.com/fruition?v=4" width="50px" alt="fruition" /></a>&nbsp;&nbsp;<a href="https://github.com/nnkostov"><img src="https://avatars.githubusercontent.com/nnkostov?v=4" width="50px" alt="nnkostov" /></a>&nbsp;&nbsp;<a href="https://github.com/morcos"><img src="https://avatars.githubusercontent.com/morcos?v=4" width="50px" alt="morcos" /></a>&nbsp;&nbsp;<a href="https://github.com/pingbotan"><img src="https://avatars.githubusercontent.com/pingbotan?v=4" width="50px" alt="pingbotan" /></a>&nbsp;&nbsp;<a href="https://github.com/maxxflyer"><img src="https://avatars.githubusercontent.com/maxxflyer?v=4" width="50px" alt="maxxflyer" /></a>&nbsp;&nbsp;<a href="https://github.com/tommi-joentakanen"><img src="https://avatars.githubusercontent.com/tommi-joentakanen?v=4" width="50px" alt="tommi-joentakanen" /></a>&nbsp;&nbsp;<a href="https://github.com/hunteraraujo"><img src="https://avatars.githubusercontent.com/hunteraraujo?v=4" width="50px" alt="hunteraraujo" /></a>&nbsp;&nbsp;<a href="https://github.com/projectonegames"><img src="https://avatars.githubusercontent.com/projectonegames?v=4" width="50px" alt="projectonegames" /></a>&nbsp;&nbsp;<a href="https://github.com/tullytim"><img src="https://avatars.githubusercontent.com/tullytim?v=4" width="50px" alt="tullytim" /></a>&nbsp;&nbsp;<a href="https://github.com/comet-ml"><img src="https://avatars.githubusercontent.com/comet-ml?v=4" width="50px" alt="comet-ml" /></a>&nbsp;&nbsp;<a href="https://github.com/thepok"><img src="https://avatars.githubusercontent.com/thepok?v=4" width="50px" alt="thepok" /></a>&nbsp;&nbsp;<a href="https://github.com/prompthero"><img src="https://avatars.githubusercontent.com/prompthero?v=4" width="50px" alt="prompthero" /></a>&nbsp;&nbsp;<a href="https://github.com/sunchongren"><img src="https://avatars.githubusercontent.com/sunchongren?v=4" width="50px" alt="sunchongren" /></a>&nbsp;&nbsp;<a href="https://github.com/neverinstall"><img src="https://avatars.githubusercontent.com/neverinstall?v=4" width="50px" alt="neverinstall" /></a>&nbsp;&nbsp;<a href="https://github.com/josephcmiller2"><img src="https://avatars.githubusercontent.com/josephcmiller2?v=4" width="50px" alt="josephcmiller2" /></a>&nbsp;&nbsp;<a href="https://github.com/yx3110"><img src="https://avatars.githubusercontent.com/yx3110?v=4" width="50px" alt="yx3110" /></a>&nbsp;&nbsp;<a href="https://github.com/MBassi91"><img src="https://avatars.githubusercontent.com/MBassi91?v=4" width="50px" alt="MBassi91" /></a>&nbsp;&nbsp;<a href="https://github.com/SpacingLily"><img src="https://avatars.githubusercontent.com/SpacingLily?v=4" width="50px" alt="SpacingLily" /></a>&nbsp;&nbsp;<a href="https://github.com/arthur-x88"><img src="https://avatars.githubusercontent.com/arthur-x88?v=4" width="50px" alt="arthur-x88" /></a>&nbsp;&nbsp;<a href="https://github.com/ciscodebs"><img src="https://avatars.githubusercontent.com/ciscodebs?v=4" width="50px" alt="ciscodebs" /></a>&nbsp;&nbsp;<a href="https://github.com/christian-gheorghe"><img src="https://avatars.githubusercontent.com/christian-gheorghe?v=4" width="50px" alt="christian-gheorghe" /></a>&nbsp;&nbsp;<a href="https://github.com/EngageStrategies"><img src="https://avatars.githubusercontent.com/EngageStrategies?v=4" width="50px" alt="EngageStrategies" /></a>&nbsp;&nbsp;<a href="https://github.com/jondwillis"><img src="https://avatars.githubusercontent.com/jondwillis?v=4" width="50px" alt="jondwillis" /></a>&nbsp;&nbsp;<a href="https://github.com/Cameron-Fulton"><img src="https://avatars.githubusercontent.com/Cameron-Fulton?v=4" width="50px" alt="Cameron-Fulton" /></a>&nbsp;&nbsp;<a href="https://github.com/AryaXAI"><img src="https://avatars.githubusercontent.com/AryaXAI?v=4" width="50px" alt="AryaXAI" /></a>&nbsp;&nbsp;<a href="https://github.com/AuroraHolding"><img src="https://avatars.githubusercontent.com/AuroraHolding?v=4" width="50px" alt="AuroraHolding" /></a>&nbsp;&nbsp;<a href="https://github.com/Mr-Bishop42"><img src="https://avatars.githubusercontent.com/Mr-Bishop42?v=4" width="50px" alt="Mr-Bishop42" /></a>&nbsp;&nbsp;<a href="https://github.com/doverhq"><img src="https://avatars.githubusercontent.com/doverhq?v=4" width="50px" alt="doverhq" /></a>&nbsp;&nbsp;<a href="https://github.com/johnculkin"><img src="https://avatars.githubusercontent.com/johnculkin?v=4" width="50px" alt="johnculkin" /></a>&nbsp;&nbsp;<a href="https://github.com/marv-technology"><img src="https://avatars.githubusercontent.com/marv-technology?v=4" width="50px" alt="marv-technology" /></a>&nbsp;&nbsp;<a href="https://github.com/ikarosai"><img src="https://avatars.githubusercontent.com/ikarosai?v=4" width="50px" alt="ikarosai" /></a>&nbsp;&nbsp;<a href="https://github.com/ColinConwell"><img src="https://avatars.githubusercontent.com/ColinConwell?v=4" width="50px" alt="ColinConwell" /></a>&nbsp;&nbsp;<a href="https://github.com/humungasaurus"><img src="https://avatars.githubusercontent.com/humungasaurus?v=4" width="50px" alt="humungasaurus" /></a>&nbsp;&nbsp;<a href="https://github.com/terpsfreak"><img src="https://avatars.githubusercontent.com/terpsfreak?v=4" width="50px" alt="terpsfreak" /></a>&nbsp;&nbsp;<a href="https://github.com/iddelacruz"><img src="https://avatars.githubusercontent.com/iddelacruz?v=4" width="50px" alt="iddelacruz" /></a>&nbsp;&nbsp;<a href="https://github.com/thisisjeffchen"><img src="https://avatars.githubusercontent.com/thisisjeffchen?v=4" width="50px" alt="thisisjeffchen" /></a>&nbsp;&nbsp;<a href="https://github.com/nicoguyon"><img src="https://avatars.githubusercontent.com/nicoguyon?v=4" width="50px" alt="nicoguyon" /></a>&nbsp;&nbsp;<a href="https://github.com/arjunb023"><img src="https://avatars.githubusercontent.com/arjunb023?v=4" width="50px" alt="arjunb023" /></a>&nbsp;&nbsp;<a href="https://github.com/Nalhos"><img src="https://avatars.githubusercontent.com/Nalhos?v=4" width="50px" alt="Nalhos" /></a>&nbsp;&nbsp;<a href="https://github.com/belharethsami"><img src="https://avatars.githubusercontent.com/belharethsami?v=4" width="50px" alt="belharethsami" /></a>&nbsp;&nbsp;<a href="https://github.com/Mobivs"><img src="https://avatars.githubusercontent.com/Mobivs?v=4" width="50px" alt="Mobivs" /></a>&nbsp;&nbsp;<a href="https://github.com/txtr99"><img src="https://avatars.githubusercontent.com/txtr99?v=4" width="50px" alt="txtr99" /></a>&nbsp;&nbsp;<a href="https://github.com/ntwrite"><img src="https://avatars.githubusercontent.com/ntwrite?v=4" width="50px" alt="ntwrite" /></a>&nbsp;&nbsp;<a href="https://github.com/founderblocks-sils"><img src="https://avatars.githubusercontent.com/founderblocks-sils?v=4" width="50px" alt="founderblocks-sils" /></a>&nbsp;&nbsp;<a href="https://github.com/kMag410"><img src="https://avatars.githubusercontent.com/kMag410?v=4" width="50px" alt="kMag410" /></a>&nbsp;&nbsp;<a href="https://github.com/angiaou"><img src="https://avatars.githubusercontent.com/angiaou?v=4" width="50px" alt="angiaou" /></a>&nbsp;&nbsp;<a href="https://github.com/garythebat"><img src="https://avatars.githubusercontent.com/garythebat?v=4" width="50px" alt="garythebat" /></a>&nbsp;&nbsp;<a href="https://github.com/lmaugustin"><img src="https://avatars.githubusercontent.com/lmaugustin?v=4" width="50px" alt="lmaugustin" /></a>&nbsp;&nbsp;<a href="https://github.com/shawnharmsen"><img src="https://avatars.githubusercontent.com/shawnharmsen?v=4" width="50px" alt="shawnharmsen" /></a>&nbsp;&nbsp;<a href="https://github.com/clortegah"><img src="https://avatars.githubusercontent.com/clortegah?v=4" width="50px" alt="clortegah" /></a>&nbsp;&nbsp;<a href="https://github.com/MetaPath01"><img src="https://avatars.githubusercontent.com/MetaPath01?v=4" width="50px" alt="MetaPath01" /></a>&nbsp;&nbsp;<a href="https://github.com/sekomike910"><img src="https://avatars.githubusercontent.com/sekomike910?v=4" width="50px" alt="sekomike910" /></a>&nbsp;&nbsp;<a href="https://github.com/MediConCenHK"><img src="https://avatars.githubusercontent.com/MediConCenHK?v=4" width="50px" alt="MediConCenHK" /></a>&nbsp;&nbsp;<a href="https://github.com/svpermari0"><img src="https://avatars.githubusercontent.com/svpermari0?v=4" width="50px" alt="svpermari0" /></a>&nbsp;&nbsp;<a href="https://github.com/jacobyoby"><img src="https://avatars.githubusercontent.com/jacobyoby?v=4" width="50px" alt="jacobyoby" /></a>&nbsp;&nbsp;<a href="https://github.com/turintech"><img src="https://avatars.githubusercontent.com/turintech?v=4" width="50px" alt="turintech" /></a>&nbsp;&nbsp;<a href="https://github.com/allenstecat"><img src="https://avatars.githubusercontent.com/allenstecat?v=4" width="50px" alt="allenstecat" /></a>&nbsp;&nbsp;<a href="https://github.com/CatsMeow492"><img src="https://avatars.githubusercontent.com/CatsMeow492?v=4" width="50px" alt="CatsMeow492" /></a>&nbsp;&nbsp;<a href="https://github.com/tommygeee"><img src="https://avatars.githubusercontent.com/tommygeee?v=4" width="50px" alt="tommygeee" /></a>&nbsp;&nbsp;<a href="https://github.com/judegomila"><img src="https://avatars.githubusercontent.com/judegomila?v=4" width="50px" alt="judegomila" /></a>&nbsp;&nbsp;<a href="https://github.com/cfarquhar"><img src="https://avatars.githubusercontent.com/cfarquhar?v=4" width="50px" alt="cfarquhar" /></a>&nbsp;&nbsp;<a href="https://github.com/ZoneSixGames"><img src="https://avatars.githubusercontent.com/ZoneSixGames?v=4" width="50px" alt="ZoneSixGames" /></a>&nbsp;&nbsp;<a href="https://github.com/kenndanielso"><img src="https://avatars.githubusercontent.com/kenndanielso?v=4" width="50px" alt="kenndanielso" /></a>&nbsp;&nbsp;<a href="https://github.com/CrypteorCapital"><img src="https://avatars.githubusercontent.com/CrypteorCapital?v=4" width="50px" alt="CrypteorCapital" /></a>&nbsp;&nbsp;<a href="https://github.com/sultanmeghji"><img src="https://avatars.githubusercontent.com/sultanmeghji?v=4" width="50px" alt="sultanmeghji" /></a>&nbsp;&nbsp;<a href="https://github.com/jenius-eagle"><img src="https://avatars.githubusercontent.com/jenius-eagle?v=4" width="50px" alt="jenius-eagle" /></a>&nbsp;&nbsp;<a href="https://github.com/josephjacks"><img src="https://avatars.githubusercontent.com/josephjacks?v=4" width="50px" alt="josephjacks" /></a>&nbsp;&nbsp;<a href="https://github.com/pingshian0131"><img src="https://avatars.githubusercontent.com/pingshian0131?v=4" width="50px" alt="pingshian0131" /></a>&nbsp;&nbsp;<a href="https://github.com/AIdevelopersAI"><img src="https://avatars.githubusercontent.com/AIdevelopersAI?v=4" width="50px" alt="AIdevelopersAI" /></a>&nbsp;&nbsp;<a href="https://github.com/ternary5"><img src="https://avatars.githubusercontent.com/ternary5?v=4" width="50px" alt="ternary5" /></a>&nbsp;&nbsp;<a href="https://github.com/ChrisDMT"><img src="https://avatars.githubusercontent.com/ChrisDMT?v=4" width="50px" alt="ChrisDMT" /></a>&nbsp;&nbsp;<a href="https://github.com/AcountoOU"><img src="https://avatars.githubusercontent.com/AcountoOU?v=4" width="50px" alt="AcountoOU" /></a>&nbsp;&nbsp;<a href="https://github.com/chatgpt-prompts"><img src="https://avatars.githubusercontent.com/chatgpt-prompts?v=4" width="50px" alt="chatgpt-prompts" /></a>&nbsp;&nbsp;<a href="https://github.com/Partender"><img src="https://avatars.githubusercontent.com/Partender?v=4" width="50px" alt="Partender" /></a>&nbsp;&nbsp;<a href="https://github.com/Daniel1357"><img src="https://avatars.githubusercontent.com/Daniel1357?v=4" width="50px" alt="Daniel1357" /></a>&nbsp;&nbsp;<a href="https://github.com/KiaArmani"><img src="https://avatars.githubusercontent.com/KiaArmani?v=4" width="50px" alt="KiaArmani" /></a>&nbsp;&nbsp;<a href="https://github.com/zkonduit"><img src="https://avatars.githubusercontent.com/zkonduit?v=4" width="50px" alt="zkonduit" /></a>&nbsp;&nbsp;<a href="https://github.com/fabrietech"><img src="https://avatars.githubusercontent.com/fabrietech?v=4" width="50px" alt="fabrietech" /></a>&nbsp;&nbsp;<a href="https://github.com/scryptedinc"><img src="https://avatars.githubusercontent.com/scryptedinc?v=4" width="50px" alt="scryptedinc" /></a>&nbsp;&nbsp;<a href="https://github.com/coreyspagnoli"><img src="https://avatars.githubusercontent.com/coreyspagnoli?v=4" width="50px" alt="coreyspagnoli" /></a>&nbsp;&nbsp;<a href="https://github.com/AntonioCiolino"><img src="https://avatars.githubusercontent.com/AntonioCiolino?v=4" width="50px" alt="AntonioCiolino" /></a>&nbsp;&nbsp;<a href="https://github.com/Dradstone"><img src="https://avatars.githubusercontent.com/Dradstone?v=4" width="50px" alt="Dradstone" /></a>&nbsp;&nbsp;<a href="https://github.com/CarmenCocoa"><img src="https://avatars.githubusercontent.com/CarmenCocoa?v=4" width="50px" alt="CarmenCocoa" /></a>&nbsp;&nbsp;<a href="https://github.com/bentoml"><img src="https://avatars.githubusercontent.com/bentoml?v=4" width="50px" alt="bentoml" /></a>&nbsp;&nbsp;<a href="https://github.com/merwanehamadi"><img src="https://avatars.githubusercontent.com/merwanehamadi?v=4" width="50px" alt="merwanehamadi" /></a>&nbsp;&nbsp;<a href="https://github.com/vkozacek"><img src="https://avatars.githubusercontent.com/vkozacek?v=4" width="50px" alt="vkozacek" /></a>&nbsp;&nbsp;<a href="https://github.com/ASmithOWL"><img src="https://avatars.githubusercontent.com/ASmithOWL?v=4" width="50px" alt="ASmithOWL" /></a>&nbsp;&nbsp;<a href="https://github.com/tekelsey"><img src="https://avatars.githubusercontent.com/tekelsey?v=4" width="50px" alt="tekelsey" /></a>&nbsp;&nbsp;<a href="https://github.com/GalaxyVideoAgency"><img src="https://avatars.githubusercontent.com/GalaxyVideoAgency?v=4" width="50px" alt="GalaxyVideoAgency" /></a>&nbsp;&nbsp;<a href="https://github.com/wenfengwang"><img src="https://avatars.githubusercontent.com/wenfengwang?v=4" width="50px" alt="wenfengwang" /></a>&nbsp;&nbsp;<a href="https://github.com/rviramontes"><img src="https://avatars.githubusercontent.com/rviramontes?v=4" width="50px" alt="rviramontes" /></a>&nbsp;&nbsp;<a href="https://github.com/indoor47"><img src="https://avatars.githubusercontent.com/indoor47?v=4" width="50px" alt="indoor47" /></a>&nbsp;&nbsp;<a href="https://github.com/ZERO-A-ONE"><img src="https://avatars.githubusercontent.com/ZERO-A-ONE?v=4" width="50px" alt="ZERO-A-ONE" /></a>&nbsp;&nbsp;</p>



## 🚀 功能

- 🌐 可以上网搜索东西
- 💾 长期和短期记忆
- 🧠 使用 GPT-4 作为文本生成模型
- 🔗 访问（国外）流行的网站
- 🗃️ 使用 GPT-3.5 进行文件存储和汇总
- 🔌 来自插件的额外功能

## Quickstart

1. 获取 OpenAI 的 [API Key](https://platform.openai.com/account/api-keys)
2. 下载 [最新的稳定版](https://github.com/Significant-Gravitas/Auto-GPT/releases/latest)
3. 查看 [安装教程][docs/setup]
4. 配置你想要的功能，或者[安装插件][docs/plugins]
5. [运行][docs/usage]

你可以在[文档][docs]中查看完整的安装、使用教程。

[docs]: https://significant-gravitas.github.io/Auto-GPT/

## 📖 文档
* [⚙️ 安装][docs/setup]
* [💻 使用][docs/usage]
* [🔌 插件][docs/plugins]
* 配置
  * [🔍 搜索](https://significant-gravitas.github.io/Auto-GPT/configuration/search/)
  * [🧠 记忆](https://significant-gravitas.github.io/Auto-GPT/configuration/memory/)
  * [🗣️ 文本转语音](https://significant-gravitas.github.io/Auto-GPT/configuration/voice/)
  * [🖼️ 图像生成](https://significant-gravitas.github.io/Auto-GPT/configuration/imagegen/)

[docs/setup]: https://significant-gravitas.github.io/Auto-GPT/setup/
[docs/usage]: https://significant-gravitas.github.io/Auto-GPT/usage/
[docs/plugins]: https://significant-gravitas.github.io/Auto-GPT/plugins/

## ⚠️ 局限性

该模型展现了 GPT-4 的一些潜力，但它不是万能的：

1. 没有最好，但是我们可以更好
2. 不要指望我们可以帮助你上班摸鱼，它在真实的业务场景中表现欠佳
3. 时刻关注你的API使用情况，不然一觉醒来，房子归OpenAI了！！！

## 🛡 免责声明（请以英文原文为准）（Google翻译）

该项目 Auto-GPT 是一个实验性应用程序，按“原样”提供，没有任何明示或暗示的保证。 使用本软件，即表示您同意承担与其使用相关的所有风险，包括但不限于数据丢失、系统故障或可能出现的任何其他问题。

本项目的开发者和贡献者对因使用本软件而可能发生的任何损失、损害或其他后果不承担任何责任或义务。 您对基于 Auto-GPT 提供的信息做出的任何决定和行动承担全部责任。

**请注意，由于使用了太多的 Token，使用 GPT-4 语言模型可能会很昂贵。**通过使用此项目，您承认您有责任监控和管理您自己的Token使用情况和相关费用。 强烈建议定期检查您的 OpenAI API 使用情况并设置任何必要的限制或警报以防止意外收费。

作为一项自主实验，Auto-GPT 可能会生成不符合现实世界商业惯例或法律要求的内容或采取的行动。 您有责任确保基于此软件的输出做出的任何行动或决定符合所有适用的法律、法规和道德标准。 本项目的开发者和贡献者对因使用本软件而产生的任何后果不承担任何责任。

通过使用 Auto-GPT，您同意就任何和所有索赔、损害、损失、责任、成本和费用（包括合理的律师费）对开发人员、贡献者和任何关联方进行赔偿、辩护并使其免受损害 因您使用本软件或您违反这些条款而引起的。

## 🐦 在推特上面找到我们

关注我们的推特账号获取最新的新闻、更新. 与开发者和 AI 的账户讨论。

- **开发者**: [@siggravitas](https://twitter.com/siggravitas) 
- **Entrepreneur-GPT**: [@En_GPT](https://twitter.com/En_GPT)

我们期待与您联系并聆听您对 Auto-GPT 的想法、想法和体验。 关注我们的 Twitter，让我们一起探索 AI！

<p align="center">
  <a href="https://star-history.com/#Torantulino/auto-gpt&Date">
    <img src="https://api.star-history.com/svg?repos=Torantulino/auto-gpt&type=Date" alt="Star History Chart">
  </a>
</p>

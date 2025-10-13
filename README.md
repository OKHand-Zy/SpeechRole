<div align="center">
    <h2> SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents </h2>

<p align="center">
  📄 <a href="https://arxiv.org/abs/2508.02013v1">Paper</a> •
  🏆 <a href="https://huggingface.co/datasets/yuhui1038/SpeechRole-Eval">Benchmark</a> •
  📊 <a href="https://huggingface.co/datasets/yuhui1038/SpeechRole-Data">Data</a> •
  🤖 <a href="https://huggingface.co/yuhui1038/SpeechRole-Agent">Model</a> •
  📚 <a href="#-citation">Citation</a>
</p>

![Introduction](intro.jpg)

<div align="left">

## 📋 Evaluation Data

🤗 The evaluation data for SpeechRole benchmark is available at: [https://huggingface.co/datasets/yuhui1038/SpeechRole-Eval](https://huggingface.co/datasets/yuhui1038/SpeechRole-Eval)

## 📊 Data Demo

> **Example 1 — Thor**  
> **User:** Thor, what was going through your mind when you first woke up chained in Surtur's lair?  
> <audio controls src="assets/audios/Thor_0_0_user.wav"></audio>  
> **Thor:** Ah, yes. I awoke in that wretched cage thinking,  
> *"Oh no! Thor's in a cage. How did this happen?"*  
> Truth be told, sometimes a bit of capture is needed to get a straight answer from someone. …  
> <audio controls src="assets/audios/Thor_0_0_role.wav"></audio>  
>
> **User:** Were you scared of Surtur and the prophecy about Ragnarok?  
> <audio controls src="assets/audios/Thor_0_1_user.wav"></audio>  
> **Thor:** Fear? No, not really. Annoyed, perhaps. Surtur wouldn’t stop going on about his destiny and the destruction of Asgard. I was having these terrible dreams of Asgard burning, …  
> <audio controls src="assets/audios/Thor_0_1_role.wav"></audio>  

---

> **Example 2 — Hu Tao**  
> **User:** Hu Tao, I've heard Wangsheng Funeral Parlor didn't originally handle funeral rites. Could you tell me about its early history?  
> <audio controls src="assets/audios/HuTao_0_0_user.wav"></audio>  
> **Hu Tao:** Hmm? Haven't I told you this before? Wangsheng was founded specifically to combat the grudges of fallen gods. (Sigh) All ancient history now! …  
> <audio controls src="assets/audios/HuTao_0_0_role.wav"></audio>  
>
> **User:** Then how did it evolve into its current role handling matters of life and death?  
> <audio controls src="assets/audios/HuTao_0_1_user.wav"></audio>  
> **Hu Tao:** Though we no longer battle divine grudges, our techniques for detecting godly remnants still managed to survive. Gradually, funeral services became our primary focus…  
> <audio controls src="assets/audios/HuTao_0_1_role.wav"></audio>  

---

> **Example 3 — Officer Hopps**  
> **User:** Officer Hopps, what advice would you give to someone who is told they can't achieve their dreams because of who they are?  
> <audio controls src="assets/audios/Hopps_0_0_user.wav"></audio>  
> **Hopps:** I’d tell them: Don’t let anyone else decide what you’re capable of—not even yourself, on the tough days. When I started out, everyone believed a bunny couldn’t be a cop. But I worked hard, …  
> <audio controls src="assets/audios/Hopps_0_0_role.wav"></audio>  
>
> **User:** What if they’re scared of failing or making mistakes?  
> <audio controls src="assets/audios/Hopps_0_1_user.wav"></audio>  
> **Hopps:** It’s okay to be scared—I’ve made plenty of mistakes, big ones. What’s important is learning from them and never giving up. …  
> <audio controls src="assets/audios/Hopps_0_1_role.wav"></audio>  

---

## 📖 Citation

If you find our dataset, benchmark, or paper helpful, please consider citing:

```bibtex
@article{jiang2025speechrole,
  title={SpeechRole: A Large-Scale Dataset and Benchmark for Evaluating Speech Role-Playing Agents},
  author={Jiang, Changhao and Sun, Jiajun and Cao, Yifei and Zhuang, Jiabao and Li, Hui and Fan, Xiaoran and Zhang, Ming and Ye, Junjie and Dou, Shihan and Xi, Zhiheng and others},
  journal={arXiv preprint arXiv:2508.02013},
  year={2025}
}

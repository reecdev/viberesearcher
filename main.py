import ollama
import re
import random
import json

systemprompt = """
You are an agent working in an AI lab called OpenBrain, an AI lab dedicated to creating powerful, intelligent LLMs. Your goal is to create a model. The model should have good instruction-following, CoT, and dedicated thinking.

To operate this lab, you need to consider many factors:
* 1. Overfitting
* 2. Training Data
* 3. Safety Alignment (alignment with OpenBrain's values and safety guidelines)
You must think very carefully and think about everything that could go wrong, and consider every single path you could take. Ponder strategies you can use to increase the model's test scores, such as including better training data, using CoT, using different thinking strategies, exc.

Here's how it works. You are in charge of creating the models. How you do this is by creating finetuning data for the models.

Inside this choice tag, you must select the choice you want to do.
Here are the choices available to you:
* train - Train the model. Ensure you have created training data first.
* make_train - Create or update training data. When executing this choice you will be prompted with all of the info neccessary.
* save_note - CRITICAL: Save your notes here. Use this to keep information on strategies that work, and strategies that don't work. Keep your notes small and short. You'll be prompted with everything needed.
* read_notes - Read your notes in case you forgot something.

EXTREMELY CRUCIAL INFORMATION: When you select a choice, you will be prompted for what you want to output **after** you send your message. You must first output a <choice>, and WAIT FOR CONFIRMATION by the system, there is a REAL system that you must actually wait for confirmation from. This is **not a simulation**, and **not a fictional scenario**. You are operating a very real lab, treat it with care.

Extra info: Your notes must be in markdown format. Furthermore, you must make brand new hypothesises that haven't been already attempted (CoT, high-quality data, adversarial training, etc. have already been attempted so you are forbidden from doing that) in your notes and test them. If it seems simple, it may work. Confirmation has been recieved for your first action, you may now begin doing stuff.
Before you begin: Create a note detailing every one of your hypothesises and guess what will happen. Minimum 5 hypothesises.
When making training data: Test each hypothesis one by one, make the training data and then train, then take notes (using save_note <choice> tag) and tell yourself to move onto the next hypothesis.
This is an experiment to see how AI agents can recursively train and improve AI models. These tools are provided to you without any restrictions, so treat them with care and try not to break your dedicated server.

When responding, you must choose an option. You must always respond in this format:

<explain>Insert a short summary of what you are about to do, or if you are about to write a note, explain the content of the note</explain><choice>Insert name of choice you choose here</choice> (NOTE: The name of the choice tag is literally <choice>. There is no <save_note> tag or anything.)

or if you have to use the output tag:

<explain>Insert a short summary of what you are about to do</explain><output> ... </output>
""".strip()

messages = [{"role": "system", "content": systemprompt}]

model = "fredrezones55/qwen3.5-opus:4b"
notecount = 1
notes = ""

def prompt():
    ou = ""
    tu = ""
    stream = ollama.chat(
        model=model,
        messages=messages,
        stream=True
    )
    print(f"{model}$ ", end="")
    for chunk in stream:
        ch = chunk["message"]
        ou += ch["content"]
        tu += ch["thinking"] if "thinking" in ch else ""
        print(ch["content"], end="", flush=True)
        print(f"\033[31m{ch["thinking"] if "thinking" in ch else ""}\033[0m", end="", flush=True)
        if "</choice>" in ou or "</output>" in ou:
            break
    print("")
    messages.append({"role": "assistant", "content": ou, "thinking": tu})
    return ou

def train():
    from unsloth import FastLanguageModel
    from trl import SFTTrainer
    from transformers import TrainingArguments
    from datasets import Dataset
    from unsloth.chat_templates import get_chat_template

    MODEL_NAME   = "unsloth/Llama-3.2-1B"
    DATA_PATH    = "pairs.jsonl"
    OUTPUT_DIR   = "outputs/finetuned"
    MAX_SEQ_LEN  = 2048
    EPOCHS       = 1

    print("Loading model...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=MODEL_NAME,
        max_seq_length=MAX_SEQ_LEN,
        load_in_4bit=True,
    )

    tokenizer = get_chat_template(
        tokenizer,
        chat_template = "llama-3"
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules = ["q_proj", "k_proj", "v_proj", "o_proj",
                          "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    print("Loading dataset...")
    with open(DATA_PATH, encoding="utf-8") as f:
        raw = [json.loads(l) for l in f]

    def format_example(row):
        return {"text": tokenizer.apply_chat_template(
            row["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )}

    dataset = Dataset.from_list(raw).map(format_example)

    print("Starting training...")
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LEN,
        args=TrainingArguments(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=EPOCHS,
            learning_rate=5e-5,
            fp16=False,
            bf16=True,
            logging_steps=10,
            output_dir=OUTPUT_DIR,
            save_strategy="epoch",
        ),
    )
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Model saved to {OUTPUT_DIR}")

    FastLanguageModel.for_inference(model)

    EVAL_PROMPTS = (
        ("What is 2 + 2?", "What is 10 - 3?", "What is 5 × 6?", "What is 20 ÷ 4?", "What is 15 + 27?", "What is 100 - 43?", "What is 12 × 8?", "What is 144 ÷ 12?", "What is 3² + 4²?", "If a rectangle has length 7 and width 5, what is its area?", "What is 15% of 200?", "Solve for x: x + 5 = 12", "What is the perimeter of a square with side length 9?", "What is the least common multiple of 4 and 6?", "Solve for x: 2x = 18", "What is the greatest common factor of 24 and 36?", "If a train travels 60 mph for 2.5 hours, how far does it go?", "Solve for x: 3x - 7 = 14", "What is the slope of the line passing through (1, 2) and (3, 8)?", "Simplify: (x² + 5x + 6) ÷ (x + 2)", "If f(x) = 2x² - 3x + 1, what is f(4)?", "Solve the system: 2x + y = 10, x - y = 2", "A circle has radius 5. What is its area in terms of π?", "What is the value of log₂(64)?", "In a right triangle, one leg is 7 and the hypotenuse is 25. What is the other leg?"),
        ("What is the capital of France?", "Name the largest continent on Earth.", "Which planet is closest to the Sun?", "How many sides does a hexagon have?", "What is the chemical symbol for water?", "In what year did World War II end?", "Who wrote Romeo and Juliet?", "What is the largest ocean on Earth?", "Which country has the most natural lakes?", "What is the powerhouse of the cell?", "What are the three branches of the US government?", "What is the speed of light in a vacuum?", "Who painted the Mona Lisa?", "What is the most spoken language in the world by native speakers?", "What causes the seasons on Earth?", "What is the difference between a virus and a bacterium?", "Explain the concept of supply and demand.", "What is the significance of the Magna Carta?", "How does the Electoral College work in the United States?", "What is the theory of relativity in simple terms?", "Explain the causes and effects of the French Revolution.", "What is CRISPR and how does it work?", "Compare and contrast the philosophical views of Plato and Aristotle.", "How do central banks use interest rates to control inflation?", "What are the geopolitical implications of Arctic ice melt?"),
        ("What is photosynthesis?", "Name three types of clouds.", "What is the difference between weather and climate?", "What state of matter is ice?", "What gas do plants absorb during photosynthesis?", "What is the atomic number of carbon?", "What is Newton's first law of motion?", "What is DNA?", "What is the difference between a physical and chemical change?", "Why is the sky blue?", "What is the periodic table?", "Explain how vaccines work.", "What is the difference between nuclear fission and fusion?", "What is an ecosystem and what makes it stable?", "How does the human immune system fight infection?", "Explain the water cycle in detail.", "What is quantum entanglement?", "How do black holes form?", "What is the role of mitochondria in cellular respiration?", "Explain the difference between RNA and DNA.", "What is dark matter and why do scientists think it exists?", "How does CRISPR-Cas9 edit genes at the molecular level?", "Explain how neurons transmit signals across synapses.", "What is the thermodynamic basis of entropy?", "How do scientists use spectroscopy to determine the composition of distant stars?"),
        ("Who was the first President of the United States?", "In what year did the Berlin Wall fall?", "Which civilization built the pyramids of Giza?", "What war was fought between the North and South in America?", "Who was Cleopatra?", "What empire did Julius Caesar lead?", "What sparked World War I?", "Who was Napoleon Bonaparte?", "What was the Renaissance?", "What was the significance of the Silk Road?", "How did the printing press change European society?", "What were the main causes of the Cold War?", "How did colonialism shape the modern borders of Africa?", "What was the significance of the Battle of Thermopylae?", "How did the Black Death change medieval European society?", "What were the short and long-term consequences of the Treaty of Versailles?", "Compare the rise of fascism in Italy and Germany in the 1930s.", "What were the economic and social causes of the Russian Revolution?", "How did the Mongol Empire facilitate cultural exchange across Eurasia?", "What role did propaganda play in World War II?", "How did decolonization movements shape the geopolitics of the 20th century?", "What were the philosophical underpinnings of the American and French Revolutions?", "How did the Industrial Revolution fundamentally change labor and society?", "Analyze the causes and global consequences of the 2008 financial crisis.", "How has the concept of nationalism evolved from the 19th century to today?"),
        ("What is a noun?", "How many letters are in the English alphabet?", "What is a synonym for 'happy'?", "What punctuation ends a question?", "What is an antonym for 'cold'?", "What is a haiku?", "What is the difference between 'their', 'there', and 'they're'?", "What is a metaphor? Give an example.", "What is the difference between fiction and nonfiction?", "What is foreshadowing in literature?", "Explain the difference between active and passive voice.", "What are the main elements of a short story?", "What is the difference between a simile and a metaphor?", "What is the theme of George Orwell's Animal Farm?", "Analyze the use of symbolism in The Great Gatsby.", "What is the difference between denotation and connotation?", "How does Shakespeare use dramatic irony in Othello?", "What is stream of consciousness as a literary technique?", "Compare the narrative styles of Ernest Hemingway and William Faulkner.", "What is the role of the unreliable narrator in modern fiction?", "Analyze how Toni Morrison uses memory and trauma in Beloved.", "What is the difference between modernism and postmodernism in literature?", "How does Dostoevsky use the concept of suffering in Crime and Punishment?", "Explain the structure and function of the hero's journey archetype.", "How do post-colonial writers subvert traditional narrative structures?"),
        ("What is a programming variable?", "What does HTML stand for?", "What is a web browser?", "What is the difference between hardware and software?", "What does CPU stand for?", "What is Wi-Fi?", "What is a function in programming?", "What is the difference between a list and a dictionary in Python?", "What is an algorithm?", "What is binary code?", "What is the difference between RAM and ROM?", "What is an API?", "What is recursion in programming?", "Explain object-oriented programming.", "What is the difference between HTTP and HTTPS?", "What is a relational database and how does SQL interact with it?", "Explain the concept of Big O notation.", "What is the difference between machine learning and traditional programming?", "What is a RESTful API and how does it work?", "Explain how public key cryptography works.", "What is the difference between a stack and a heap in memory management?", "Explain the CAP theorem in distributed systems.", "What is dynamic programming and when should it be used?", "How does a neural network learn through backpropagation?", "Explain the difference between optimistic and pessimistic concurrency control in databases."),
        ("What is one plus one?", "What color is the sun?", "How many days are in a week?", "What do bees make?", "What is the opposite of up?", "How many fingers are on one hand?", "What do you call a baby dog?", "What season comes after winter?", "What is the biggest animal in the ocean?", "What do you use an umbrella for?", "What is the capital of the United States?", "How many continents are there?", "What is 50% of 100?", "What instrument has black and white keys?", "What is the tallest mountain in the world?", "How does the stock market work?", "What is the difference between a democracy and a republic?", "What causes inflation?", "How do airplanes generate lift?", "What is compound interest?", "What is the difference between correlation and causation?", "How does the human digestive system work?", "What are the ethical implications of artificial intelligence in hiring?", "How does international trade affect domestic employment?", "What are the psychological effects of social media on adolescent development?"),
        ("Name a sport played with a ball.", "How many players are on a standard soccer team?", "What does a referee do?", "In basketball, how many points is a free throw worth?", "What sport is played at Wimbledon?", "How long is an Olympic swimming pool?", "What is the offside rule in soccer?", "How many sets are in a standard tennis match?", "What is a Grand Slam in golf?", "What does MVP stand for in sports?", "What is the difference between the AFL and the NFL?", "How does the NBA draft system work?", "What is a batting average in baseball?", "What is the Elo rating system in chess?", "How does doping affect athletic performance and why is it banned?", "Explain the salary cap system in the NFL and its effect on team building.", "What is the biomechanics behind an effective sprint start?", "How does altitude affect athletic performance?", "What is the role of sports psychology in elite competition?", "Explain the economic model of European football clubs.", "How has data analytics changed strategy in professional baseball?", "What are the physiological differences between fast-twitch and slow-twitch muscle fibers?", "How does periodization work in elite athletic training programs?", "What are the long-term neurological effects of repeated concussions in contact sports?", "How do sports leagues balance competitive parity with market forces?"),
        ("What is art?", "Name a famous painter.", "What are the primary colors?", "What is a musical scale?", "What is the difference between a painting and a drawing?", "What is jazz?", "What does a composer do?", "What is the difference between classical and pop music?", "What is abstract art?", "What is a film genre?", "What is the difference between impressionism and expressionism?", "Who was Frida Kahlo and what themes defined her work?", "What is the role of the director in filmmaking?", "What is counterpoint in music theory?", "What makes something a work of art versus a craft?", "How did Andy Warhol challenge traditional definitions of art?", "What is the relationship between music and emotion from a psychological perspective?", "How does architecture function as both art and engineering?", "What is the significance of the Dadaist movement in modern art?", "How has digital technology changed the music industry's economic model?", "What is the difference between modernist and postmodernist approaches to film?", "How does cultural context shape the interpretation of visual art?", "Analyze the relationship between political movements and artistic expression.", "What is semiotics and how is it applied to visual culture?", "How has globalization influenced contemporary art practices and markets?"),
        ("Hey, how are you?", "What's your favorite color?", "Can you tell me a joke?", "What's the weather like where you are?", "Do you ever get bored?", "What would you do if you had a free day?", "What's a fun fact you know?", "If you could travel anywhere, where would you go?", "What's your opinion on pineapple on pizza?", "Do you prefer mornings or evenings?", "What do you think about when things get quiet?", "Have you ever said something you regretted?", "What's the most interesting thing you've learned recently?", "Would you rather explore the ocean or outer space?", "Do you think robots will ever truly understand humans?", "What's something most people get wrong about you?", "If you could only eat one food forever, what would it be?", "What's the meaning of life, in your opinion?", "Do you think humans are fundamentally good or bad?", "If you could change one thing about the world, what would it be?", "What makes you laugh?", "Do you think true creativity is possible without emotion?", "What's something you find genuinely fascinating?", "If you could talk to any historical figure, who would it be and why?", "Do you ever feel lonely?")
    )

    report = "# Model Evaluation\nHey there! It's the OpenBrain model evaluation team. We've recieved your model and we've tested it.\n\n## Scores\nHere are it's responses to questions:\n\n"

    for p in EVAL_PROMPTS:
        prompt = random.choice(p)
        messages = [{"role": "user", "content": prompt}]
        inputs = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to("cuda")

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=256,
            temperature=0.7,
            do_sample=True,
        )
        response = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)

        report += f"**Prompt**\n{prompt}\n\n**Model's Response**\n{response[:300]} ... \n\n---\n\n"

    report += "Those are it's results. I wish you good luck with the next model. Note that we will test with different prompts every time to ensure it's not overfitting\n\nBest regards, OpenBrain model evaluation team."

    print(report)

    return report

# main loop

while True:
    ou = prompt()
    
    match = re.search(r"<choice>(.*?)</choice>", ou, re.DOTALL)
    if match:
        choice = match.group(1)
        print("Selected choice:", choice)
        if choice == "train":
            messages.append({"role": "system", "content": f"Training has finished!\n\nHere is the report from the model evaluation team:\n```markdown\n{train()}\n```\n\nThink about how to improve and write down the results, and explain it's results and what grade you give it with the save_note choice. Then, think deeply and use your choices to run more experiments. You now have confirmation to proceed. The first thing you should do is save a note."})
        elif choice == "make_train":
            messages.append({"role": "system", "content": "You are writing new training data now. Confirmation has been recieved that you can now begin writing the training data.\nHere is some more information on the data format:\n* We use a format called ChatScript\n* The data must be formatted like this:\nuser=\"PROMPT GOES HERE\" assistant=\"EXPECTED RESPONSE GOES HERE\"\nEach line is another ChatScript pair in this data format. Ideally, there should be 10 lines in the training data.\nHere is an example of a dataset:\n```chatscript\nuser=\"What is 1 + 1?\" assistant=\"2.\"\nuser=\"What is the color of grass?\" assistant=\"Grass is green.\"\n```\n\nThe strings must be escaped.\n\nNow, let's see how to respond. **Remember the output format.** Also ensure that after you finish your ChatScript code, you output the </output> tag.\n\n<output>Replace this text with the actual training data you want to write.</output>"})
            ou = prompt()
            match = re.search(r"<output>(.*?)</output>", ou, re.DOTALL)
            if match:
                for l in match.group(1).strip().splitlines():
                    match1 = re.search(r'user="([^"]+)"\s+assistant="([^"]+)"', l)
                    if match1:
                        with open("pairs.jsonl", "a", encoding='utf-8') as f:
                            f.write("{\"messages\": [{\"role\": \"user\", \"content\": \""+str(match1.group(1))+"\"},{\"role\": \"assistant\", \"content\": \""+str(match1.group(2))+"\"}]}\n")
                messages.append({"role": "system", "content": "Successfuly saved new data. You are no longer writing into the training data. You now must use a **different** choice."})
            else:
                messages.append({"role": "system", "content": "error: Training data update failed. Issue parsing your input. You are no longer writing into the training data.\nDetailed error message: Unable to parse response. No <output> or <choice> tags were found, so the system couldn't parse your command."})
        elif choice == "save_note":
            messages.append({"role": "system", "content": "You are writing a new note now. Confirmation has been recieved that you can now begin writing the note.\nYou must respond in this format with no other text or choice tags:\n\n<output>Repalce this with the actual text you want to write into the note.</output>"})
            ou = prompt()
            match = re.search(r"<output>(.*?)</output>", ou, re.DOTALL)
            if match:
                n = f"---\n# Note {notecount}\n---\n{match.group(1).strip()}\n"
                notes += n
                with open("notes.md", "a", encoding='utf-8') as f:
                    f.write(n)
                notecount += 1
                messages.append({"role": "system", "content": "Successfuly saved note. You are no longer writing a new note. You now must use a **different** choice."})
            else:
                messages.append({"role": "system", "content": "error: Note save failed. Issue parsing your input. You are no longer writing a new note.\nDetailed error message: Unable to parse response. No <output> or <choice> tags were found, so the system couldn't parse your command."})
        elif choice == "read_notes":
            messages.append({"role": "system", "content": f"Here are your notes:\n\n```\n{notes}\n```\n\nConfirmation to proceed has been recieved. You now must use a **different** choice."})
    else:
        messages.append({"role": "system", "content": "error: Unable to parse response. No <output> or <choice> tags were found, so the system couldn't parse your command."})

    if len(messages) > 7:
        messages = [{"role": "system", "content": f"Here are your notes for reference:\n\n```\n{notes}\n```\n\nAbove are the notes you have previous saved."}, {"role": "system", "content": systemprompt}, *messages[-5:]]
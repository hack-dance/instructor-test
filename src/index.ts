import { createSchemaFunction } from "@/oai/fns/schema"
import OpenAI from "openai"
import { ChatCompletionCreateParamsNonStreaming } from "openai/resources/index.mjs"
import { z, ZodSchema } from "zod"

class Instruct {
  private client: OpenAI
  private mode: string

  constructor({ client, mode }: { client: OpenAI; mode: string }) {
    this.client = client
    this.mode = mode
  }

  private chatCompletion = async ({
    response_model,
    max_retries = 3,
    ...params
  }: ChatCompletionCreateParamsNonStreaming & {
    response_model: ZodSchema<unknown>
    max_retries?: number
  }) => {
    let attempts = 0

    const functionConfig = this.generateSchemaFunction({
      schema: response_model
    })

    const makeCompletionCall = async () => {
      try {
        const completion = await this.client.chat.completions.create({
          stream: false,
          ...params,
          ...functionConfig
        })

        const response = completion.choices?.[0]?.message?.function_call?.arguments ?? {}

        return response
      } catch (error) {
        throw error
      }
    }

    const makeCompletionCallWithRetries = async () => {
      try {
        return await makeCompletionCall()
      } catch (error) {
        if (attempts < max_retries) {
          attempts++
          return await makeCompletionCallWithRetries()
        } else {
          throw error
        }
      }
    }

    return await makeCompletionCallWithRetries()
  }

  private generateSchemaFunction({ schema }) {
    const { definition } = createSchemaFunction({ schema })

    return {
      function_call: {
        name: definition.name
      },
      functions: [
        {
          name: definition.name,
          description: definition.description,
          parameters: {
            type: "object",
            properties: definition.parameters,
            required: definition.required
          }
        }
      ]
    }
  }

  public chat = {
    completions: {
      create: this.chatCompletion
    }
  }
}

//----------------------------------------------//

const UserSchema = z.object({
  age: z.number(),
  name: z.string().refine(name => name.includes(" "), {
    message: "Name must contain a space"
  })
})

type User = z.infer<typeof UserSchema>

const oai = new OpenAI({
  apiKey: process.env.OPENAI_API_KEY ?? undefined,
  organization: process.env.OPENAI_ORG_ID ?? undefined
})

const client = new Instruct({
  client: oai,
  mode: ""
})

const user: User = await client.chat.completions.create({
  messages: [{ role: "user", content: "Jason Liu is 30 years old" }],
  model: "gpt-3.5-turbo",
  response_model: UserSchema,
  max_retries: 3
})

console.log(user)

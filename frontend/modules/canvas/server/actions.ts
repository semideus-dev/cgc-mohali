"use server";

import { revalidatePath } from "next/cache";
import { headers } from "next/headers";
import { eq, and, desc } from "drizzle-orm";
import { z } from "zod";

import { auth } from "@/lib/auth";
import { db } from "@/lib/db";
import { canvas } from "@/lib/db/schema";

// Validation schemas
const createCanvasSchema = z.object({
  name: z.string().min(1, "Name is required").max(255, "Name is too long"),
  prompt: z.string().min(1, "Prompt is required"),
  url: z.string().url().optional(),
});

const updateCanvasSchema = z.object({
  id: z.string().min(1, "Canvas ID is required"),
  name: z
    .string()
    .min(1, "Name is required")
    .max(255, "Name is too long")
    .optional(),
  prompt: z.string().min(1, "Prompt is required").optional(),
  url: z.string().url().optional(),
});

// Types for our actions
export type CreateCanvasInput = z.infer<typeof createCanvasSchema>;
export type UpdateCanvasInput = z.infer<typeof updateCanvasSchema>;

// Helper function to get current user
async function getCurrentUser() {
  const session = await auth.api.getSession({
    headers: await headers(),
  });

  if (!session) {
    throw new Error("Unauthorized");
  }

  return session.user;
}

// Create canvas
export async function createCanvas(input: CreateCanvasInput) {
  try {
    const user = await getCurrentUser();
    const validatedInput = createCanvasSchema.parse(input);

    const newCanvas = await db
      .insert(canvas)
      .values({
        id: crypto.randomUUID(),
        name: validatedInput.name,
        prompt: validatedInput.prompt,
        url: validatedInput.url,
        userId: user.id,
        createdAt: new Date(),
        updatedAt: new Date(),
      })
      .returning()
      .then((rows) => rows[0]);

    revalidatePath("/dashboard");
    return { success: true, data: newCanvas };
  } catch (error) {
    console.error("Error creating canvas:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to create canvas",
    };
  }
}

// Get all canvases for current user
export async function getCanvases() {
  try {
    const user = await getCurrentUser();

    const canvases = await db
      .select()
      .from(canvas)
      .where(eq(canvas.userId, user.id))
      .orderBy(desc(canvas.updatedAt));

    return { success: true, data: canvases };
  } catch (error) {
    console.error("Error fetching canvases:", error);
    return {
      success: false,
      error:
        error instanceof Error ? error.message : "Failed to fetch canvases",
    };
  }
}

// Get single canvas by ID
export async function getCanvas(id: string) {
  try {
    const user = await getCurrentUser();

    const canvasItem = await db
      .select()
      .from(canvas)
      .where(and(eq(canvas.id, id), eq(canvas.userId, user.id)))
      .then((rows) => rows[0]);

    if (!canvasItem) {
      return { success: false, error: "Canvas not found" };
    }

    return { success: true, data: canvasItem };
  } catch (error) {
    console.error("Error fetching canvas:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to fetch canvas",
    };
  }
}

// Update canvas
export async function updateCanvas(input: UpdateCanvasInput) {
  try {
    const user = await getCurrentUser();
    const validatedInput = updateCanvasSchema.parse(input);

    // Check if canvas exists and belongs to user
    const existingCanvas = await db
      .select()
      .from(canvas)
      .where(and(eq(canvas.id, validatedInput.id), eq(canvas.userId, user.id)))
      .then((rows) => rows[0]);

    if (!existingCanvas) {
      return { success: false, error: "Canvas not found" };
    }

    // Prepare update data
    const updateData: Partial<typeof canvas.$inferInsert> = {
      updatedAt: new Date(),
    };

    if (validatedInput.name !== undefined) {
      updateData.name = validatedInput.name;
    }
    if (validatedInput.prompt !== undefined) {
      updateData.prompt = validatedInput.prompt;
    }
    if (validatedInput.url !== undefined) {
      updateData.url = validatedInput.url;
    }

    const updatedCanvas = await db
      .update(canvas)
      .set(updateData)
      .where(eq(canvas.id, validatedInput.id))
      .returning()
      .then((rows) => rows[0]);

    revalidatePath("/dashboard");
    return { success: true, data: updatedCanvas };
  } catch (error) {
    console.error("Error updating canvas:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to update canvas",
    };
  }
}

// Delete canvas
export async function deleteCanvas(id: string) {
  try {
    const user = await getCurrentUser();

    // Check if canvas exists and belongs to user
    const existingCanvas = await db
      .select()
      .from(canvas)
      .where(and(eq(canvas.id, id), eq(canvas.userId, user.id)))
      .then((rows) => rows[0]);

    if (!existingCanvas) {
      return { success: false, error: "Canvas not found" };
    }

    await db.delete(canvas).where(eq(canvas.id, id));

    revalidatePath("/dashboard");
    return { success: true, data: { id } };
  } catch (error) {
    console.error("Error deleting canvas:", error);
    return {
      success: false,
      error: error instanceof Error ? error.message : "Failed to delete canvas",
    };
  }
}

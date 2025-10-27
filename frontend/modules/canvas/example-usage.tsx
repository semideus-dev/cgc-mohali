"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Textarea } from "@/components/ui/textarea";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Skeleton } from "@/components/ui/skeleton";
import {
  useCanvases,
  useCreateCanvas,
  useUpdateCanvas,
  useDeleteCanvas,
  type Canvas,
} from "./server/hooks";

// Example component showing how to use the canvas hooks
export function CanvasManager() {
  const [isCreating, setIsCreating] = useState(false);
  const [editingCanvas, setEditingCanvas] = useState<Canvas | null>(null);

  // Fetch all canvases
  const { data: canvases, isLoading, error } = useCanvases();

  // Mutations
  const createCanvasMutation = useCreateCanvas();
  const updateCanvasMutation = useUpdateCanvas();
  const deleteCanvasMutation = useDeleteCanvas();

  // Handle create canvas
  const handleCreate = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    const formData = new FormData(e.currentTarget);

    const canvasData = {
      name: formData.get("name") as string,
      prompt: formData.get("prompt") as string,
      url: (formData.get("url") as string) || undefined,
    };

    try {
      await createCanvasMutation.mutateAsync(canvasData);
      setIsCreating(false);
      (e.target as HTMLFormElement).reset();
    } catch (error) {
      // Error is handled by the hook with toast
      console.error("Failed to create canvas:", error);
    }
  };

  // Handle update canvas
  const handleUpdate = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!editingCanvas) return;

    const formData = new FormData(e.currentTarget);

    const updateData = {
      id: editingCanvas.id,
      name: formData.get("name") as string,
      prompt: formData.get("prompt") as string,
      url: (formData.get("url") as string) || undefined,
    };

    try {
      await updateCanvasMutation.mutateAsync(updateData);
      setEditingCanvas(null);
    } catch (error) {
      // Error is handled by the hook with toast
      console.error("Failed to update canvas:", error);
    }
  };

  // Handle delete canvas
  const handleDelete = async (id: string) => {
    if (!confirm("Are you sure you want to delete this canvas?")) return;

    try {
      await deleteCanvasMutation.mutateAsync(id);
    } catch (error) {
      // Error is handled by the hook with toast
      console.error("Failed to delete canvas:", error);
    }
  };

  if (isLoading) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-8 w-48" />
        <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
          {Array.from({ length: 6 }).map((_, i) => (
            <Card key={i}>
              <CardHeader>
                <Skeleton className="h-6 w-3/4" />
              </CardHeader>
              <CardContent>
                <Skeleton className="h-20 w-full mb-4" />
                <div className="flex gap-2">
                  <Skeleton className="h-8 w-16" />
                  <Skeleton className="h-8 w-16" />
                </div>
              </CardContent>
            </Card>
          ))}
        </div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="text-center py-8">
        <p className="text-red-500">Error loading canvases: {error.message}</p>
        <Button onClick={() => window.location.reload()} className="mt-4">
          Retry
        </Button>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <h1 className="text-3xl font-bold">My Canvases</h1>
        <Button onClick={() => setIsCreating(true)}>Create New Canvas</Button>
      </div>

      {/* Create Form */}
      {isCreating && (
        <Card>
          <CardHeader>
            <CardTitle>Create New Canvas</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleCreate} className="space-y-4">
              <div>
                <Input
                  name="name"
                  placeholder="Canvas name"
                  required
                  disabled={createCanvasMutation.isPending}
                />
              </div>
              <div>
                <Textarea
                  name="prompt"
                  placeholder="Canvas prompt"
                  required
                  disabled={createCanvasMutation.isPending}
                />
              </div>
              <div>
                <Input
                  name="url"
                  type="url"
                  placeholder="Canvas URL (optional)"
                  disabled={createCanvasMutation.isPending}
                />
              </div>
              <div className="flex gap-2">
                <Button type="submit" disabled={createCanvasMutation.isPending}>
                  {createCanvasMutation.isPending ? "Creating..." : "Create"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setIsCreating(false)}
                  disabled={createCanvasMutation.isPending}
                >
                  Cancel
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Edit Form */}
      {editingCanvas && (
        <Card>
          <CardHeader>
            <CardTitle>Edit Canvas</CardTitle>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleUpdate} className="space-y-4">
              <div>
                <Input
                  name="name"
                  defaultValue={editingCanvas.name}
                  placeholder="Canvas name"
                  required
                  disabled={updateCanvasMutation.isPending}
                />
              </div>
              <div>
                <Textarea
                  name="prompt"
                  defaultValue={editingCanvas.prompt}
                  placeholder="Canvas prompt"
                  required
                  disabled={updateCanvasMutation.isPending}
                />
              </div>
              <div>
                <Input
                  name="url"
                  type="url"
                  defaultValue={editingCanvas.url || ""}
                  placeholder="Canvas URL (optional)"
                  disabled={updateCanvasMutation.isPending}
                />
              </div>
              <div className="flex gap-2">
                <Button type="submit" disabled={updateCanvasMutation.isPending}>
                  {updateCanvasMutation.isPending ? "Updating..." : "Update"}
                </Button>
                <Button
                  type="button"
                  variant="outline"
                  onClick={() => setEditingCanvas(null)}
                  disabled={updateCanvasMutation.isPending}
                >
                  Cancel
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Canvas List */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-3">
        {canvases?.map((canvas) => (
          <Card key={canvas.id}>
            <CardHeader>
              <CardTitle className="text-lg">{canvas.name}</CardTitle>
            </CardHeader>
            <CardContent>
              <p className="text-sm text-muted-foreground mb-4 line-clamp-3">
                {canvas.prompt}
              </p>
              {canvas.url && (
                <p className="text-xs text-blue-500 mb-4 truncate">
                  {canvas.url}
                </p>
              )}
              <div className="flex gap-2">
                <Button
                  size="sm"
                  variant="outline"
                  onClick={() => setEditingCanvas(canvas)}
                  disabled={updateCanvasMutation.isPending}
                >
                  Edit
                </Button>
                <Button
                  size="sm"
                  variant="destructive"
                  onClick={() => handleDelete(canvas.id)}
                  disabled={deleteCanvasMutation.isPending}
                >
                  {deleteCanvasMutation.isPending ? "..." : "Delete"}
                </Button>
              </div>
              <div className="text-xs text-muted-foreground mt-2">
                Created: {new Date(canvas.createdAt).toLocaleDateString()}
              </div>
            </CardContent>
          </Card>
        ))}
      </div>

      {canvases?.length === 0 && (
        <div className="text-center py-8">
          <p className="text-muted-foreground">No canvases found</p>
          <Button onClick={() => setIsCreating(true)} className="mt-4">
            Create your first canvas
          </Button>
        </div>
      )}
    </div>
  );
}

// Example hook for a single canvas component
export function useCanvasOperations() {
  const createCanvas = useCreateCanvas();
  const updateCanvas = useUpdateCanvas();
  const deleteCanvas = useDeleteCanvas();

  const operations = {
    // Create with optimistic UI
    create: async (data: { name: string; prompt: string; url?: string }) => {
      return createCanvas.mutateAsync(data);
    },

    // Update with optimistic UI
    update: async (data: {
      id: string;
      name?: string;
      prompt?: string;
      url?: string;
    }) => {
      return updateCanvas.mutateAsync(data);
    },

    // Delete with optimistic UI
    delete: async (id: string) => {
      return deleteCanvas.mutateAsync(id);
    },

    // Status flags
    isLoading:
      createCanvas.isPending ||
      updateCanvas.isPending ||
      deleteCanvas.isPending,
    isCreating: createCanvas.isPending,
    isUpdating: updateCanvas.isPending,
    isDeleting: deleteCanvas.isPending,
  };

  return operations;
}

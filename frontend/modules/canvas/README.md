# Canvas CRUD Module

This module provides complete CRUD operations for canvas management with optimistic UI updates using TanStack Query.

## Features

- ✅ **Server Actions**: Type-safe server actions for all CRUD operations
- ✅ **Optimistic UI**: Immediate UI updates with automatic rollback on errors
- ✅ **Error Handling**: Comprehensive error handling with user-friendly messages
- ✅ **Type Safety**: Full TypeScript support with proper type inference
- ✅ **Caching**: Intelligent caching and invalidation strategies
- ✅ **Toast Notifications**: Success and error notifications using Sonner

## Structure

```
modules/canvas/
├── server/
│   ├── actions.ts     # Server actions for CRUD operations
│   └── hooks.ts       # TanStack Query hooks with optimistic UI
└── example-usage.tsx  # Example component demonstrating usage
```

## Server Actions (`actions.ts`)

### Available Actions

- `createCanvas(input)` - Create a new canvas
- `getCanvases()` - Get all canvases for current user
- `getCanvas(id)` - Get single canvas by ID
- `updateCanvas(input)` - Update existing canvas
- `deleteCanvas(id)` - Delete canvas

### Input Types

```typescript
type CreateCanvasInput = {
  name: string;
  prompt: string;
  url?: string;
};

type UpdateCanvasInput = {
  id: string;
  name?: string;
  prompt?: string;
  url?: string;
};
```

### Response Format

All actions return a consistent response format:

```typescript
{
  success: boolean;
  data?: Canvas; // or Canvas[] for getCanvases
  error?: string;
}
```

## Hooks (`hooks.ts`)

### Query Hooks

- `useCanvases()` - Fetch all canvases with caching
- `useCanvas(id)` - Fetch single canvas with caching

### Mutation Hooks

- `useCreateCanvas()` - Create canvas with optimistic UI
- `useUpdateCanvas()` - Update canvas with optimistic UI
- `useDeleteCanvas()` - Delete canvas with optimistic UI

### Utility Hooks

- `useInvalidateCanvases()` - Manual cache invalidation utilities

## Usage Examples

### Basic Component

```tsx
import {
  useCanvases,
  useCreateCanvas,
  useUpdateCanvas,
  useDeleteCanvas,
} from "@/modules/canvas/server/hooks";

function CanvasManager() {
  const { data: canvases, isLoading } = useCanvases();
  const createCanvas = useCreateCanvas();
  const updateCanvas = useUpdateCanvas();
  const deleteCanvas = useDeleteCanvas();

  const handleCreate = async () => {
    await createCanvas.mutateAsync({
      name: "My Canvas",
      prompt: "Create something amazing",
      url: "https://example.com",
    });
  };

  const handleUpdate = async (id: string) => {
    await updateCanvas.mutateAsync({
      id,
      name: "Updated Canvas Name",
    });
  };

  const handleDelete = async (id: string) => {
    await deleteCanvas.mutateAsync(id);
  };

  if (isLoading) return <div>Loading...</div>;

  return (
    <div>
      {canvases?.map((canvas) => (
        <div key={canvas.id}>
          <h3>{canvas.name}</h3>
          <p>{canvas.prompt}</p>
          <button onClick={() => handleUpdate(canvas.id)}>Edit</button>
          <button onClick={() => handleDelete(canvas.id)}>Delete</button>
        </div>
      ))}
      <button onClick={handleCreate}>Create New</button>
    </div>
  );
}
```

### With Form Handling

```tsx
import { useForm } from "react-hook-form";
import { useCreateCanvas } from "@/modules/canvas/server/hooks";

function CreateCanvasForm() {
  const { register, handleSubmit, reset } = useForm();
  const createCanvas = useCreateCanvas();

  const onSubmit = async (data) => {
    try {
      await createCanvas.mutateAsync(data);
      reset(); // Clear form on success
    } catch (error) {
      // Error is automatically handled by the hook
    }
  };

  return (
    <form onSubmit={handleSubmit(onSubmit)}>
      <input
        {...register("name", { required: true })}
        placeholder="Canvas name"
      />
      <textarea
        {...register("prompt", { required: true })}
        placeholder="Prompt"
      />
      <input {...register("url")} placeholder="URL (optional)" />
      <button type="submit" disabled={createCanvas.isPending}>
        {createCanvas.isPending ? "Creating..." : "Create Canvas"}
      </button>
    </form>
  );
}
```

## Optimistic UI Features

### Create Canvas

- Immediately adds the new canvas to the UI
- Shows optimistic data while request is in flight
- Replaces with server data on success
- Rolls back on error with toast notification

### Update Canvas

- Immediately reflects changes in the UI
- Updates both list and detail views
- Reverts changes on error

### Delete Canvas

- Immediately removes canvas from UI
- Restores canvas on error

## Query Key Management

The module uses a structured query key system:

```typescript
const canvasKeys = {
  all: ["canvases"],
  lists: () => [...canvasKeys.all, "list"],
  list: (filters?) => [...canvasKeys.lists(), { filters }],
  details: () => [...canvasKeys.all, "detail"],
  detail: (id) => [...canvasKeys.details(), id],
};
```

This allows for:

- Granular cache invalidation
- Efficient data fetching
- Automatic background refetching
- Optimistic updates

## Error Handling

- **Validation Errors**: Zod schema validation on server actions
- **Authentication**: Automatic user authentication checks
- **Database Errors**: Proper error catching and user-friendly messages
- **Network Errors**: Automatic retry and error recovery
- **Toast Notifications**: User feedback for all operations

## Authentication

All server actions automatically:

- Check for valid user session
- Return appropriate errors for unauthenticated requests
- Include user ID in database operations for data isolation

## Database Schema

The canvas table structure:

```sql
CREATE TABLE canvas (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  url TEXT,
  prompt TEXT NOT NULL,
  user_id TEXT NOT NULL REFERENCES user(id) ON DELETE CASCADE,
  created_at TIMESTAMP NOT NULL DEFAULT NOW(),
  updated_at TIMESTAMP NOT NULL DEFAULT NOW()
);
```

## Dependencies

- `@tanstack/react-query` - Data fetching and caching
- `drizzle-orm` - Database ORM
- `zod` - Schema validation
- `sonner` - Toast notifications
- `better-auth` - Authentication
- `next/cache` - Cache revalidation

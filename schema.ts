export type LogsType = {
    machine_id?: string;
    logs: string;
    timestamp?: number;
}[];

import type { CivitaiModelResponse } from "./types/civitai";

import {
    type DependencyGraphType,
} from "comfyui-json";
import { type InferSelectModel, relations } from "drizzle-orm";
import {
    type AnyPgColumn,
    boolean,
    integer,
    jsonb,
    pgEnum,
    pgSchema,
    real,
    text,
    timestamp,
    uuid,
    index,
} from "drizzle-orm/pg-core";
import { createInsertSchema, createSelectSchema } from "drizzle-zod";
import { z } from "zod";
export const dbSchema = pgSchema("comfyui_deploy");

export const usersTable = dbSchema.table("users", {
    id: text("id").primaryKey().notNull(),
    username: text("username").notNull(),
    name: text("name").notNull(),
    created_at: timestamp("created_at").defaultNow(),
    updated_at: timestamp("updated_at").defaultNow(),
});

export const workflowTable = dbSchema.table("workflows", {
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    user_id: text("user_id")
        .references(() => usersTable.id, {
            onDelete: "cascade",
        })
        .notNull(),
    org_id: text("org_id"),
    name: text("name").notNull(),
    selected_machine_id: uuid("selected_machine_id").references(
        () => machinesTable.id,
        {
            onDelete: "set null",
        },
    ),
    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
    pinned: boolean("pinned").default(false).notNull(),
});

export const workflowSchema = createSelectSchema(workflowTable);

export const workflowRelations = relations(workflowTable, ({ many, one }) => ({
    user: one(usersTable, {
        fields: [workflowTable.user_id],
        references: [usersTable.id],
    }),
    versions: many(workflowVersionTable),
    deployments: many(deploymentsTable),
    selected_machine: one(machinesTable, {
        fields: [workflowTable.selected_machine_id],
        references: [machinesTable.id],
    }),
    machine: one(machinesTable, {
        fields: [workflowTable.id],
        references: [machinesTable.id],
    }),
    runs: many(workflowRunsTable),
}));

export const workflowType = z.any();
export const workflowTypeRaw = z.object({
    last_node_id: z.number(),
    last_link_id: z.number(),
    nodes: z.array(
        z.object({
            id: z.number(),
            type: z.string(),
            widgets_values: z.array(z.any()),
        })
    ),
});

export const workflowAPINodeType = z.object({
    inputs: z.record(z.any()),
    class_type: z.string().optional(),
});

export const workflowAPIType = z.record(workflowAPINodeType);

export const workflowVersionTable = dbSchema.table("workflow_versions", {
    workflow_id: uuid("workflow_id")
        .notNull()
        .references(() => workflowTable.id, {
            onDelete: "cascade",
        }),
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    workflow: jsonb("workflow").$type<z.infer<typeof workflowType>>(),
    workflow_api: jsonb("workflow_api").$type<z.infer<typeof workflowAPIType>>(),
    user_id: text("user_id").references(() => usersTable.id, {
        onDelete: "set null",
    }),
    comment: text("comment"),
    version: integer("version").notNull(),
    snapshot: jsonb("snapshot").$type<z.infer<typeof snapshotType>>(),
    dependencies:
        jsonb("dependencies").$type<z.infer<typeof DependencyGraphType>>(),

    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
}, (table) => {
    return {
        idx_workflow_version_workflow_id: index(
            "idx_workflow_version_workflow_id",
        ).on(table.workflow_id),
    };
},);

export const CustomNodesDepsType = z.record(
    z.object({
        name: z.string(),
        node: z.array(workflowAPINodeType).optional(),
        hash: z.string().optional(),
        url: z.string(),
        files: z.array(z.string()).optional(),
        install_type: z
            .union([z.enum(["copy", "unzip", "git-clone"]), z.string()])
            .optional(),
        warning: z.string().optional(),
        pip: z.array(z.string()).optional(),
    }),
);

// export const CustomNodesDepsType = z.record(
//   z.object({
//     name: z.string(),
//     node: z.array(workflowAPINodeType).optional(),
//     hash: z.string().optional(),
//     url: z.string(),
//     pip: z.array(z.string()).optional(),
//     warning: z.string().optional(),
//   }),
// );

export const FileReferenceType = z.object({
    name: z.string(),
    hash: z.string().optional(),
    url: z.string().optional(),
});

export const FileReferencesType = z.record(z.array(FileReferenceType));

// export const DependencyGraphType = z.object({
//   comfyui: z.string(),
//   missing_nodes: z.array(z.string()).optional(),
//   custom_nodes: CustomNodesDepsType, // Replace z.any() with the specific type if known
//   models: FileReferencesType, // Replace z.any() with the specific type if known
//   files: FileReferencesType, // Replace z.any() with the specific type if known
// });

export const workflowVersionSchema = createSelectSchema(workflowVersionTable);

export const workflowVersionRelations = relations(
    workflowVersionTable,
    ({ one }) => ({
        workflow: one(workflowTable, {
            fields: [workflowVersionTable.workflow_id],
            references: [workflowTable.id],
        }),
        user: one(usersTable, {
            fields: [workflowVersionTable.user_id],
            references: [usersTable.id],
        }),
    }),
);

export const workflowRunStatus = pgEnum("workflow_run_status", [
    "not-started",
    "running",
    "uploading",
    "success",
    "failed",
    "started",
    "queued",
    "timeout",
    "cancelled",
]);

export const webhook_status = pgEnum("webhook_status", [
    "success",
    "failed",
    "not-started",
    "running",
]);

export const deploymentEnvironment = pgEnum("deployment_environment", [
    "staging",
    "production",
    "public-share",
    "private-share",
]);

export const workflowRunOrigin = pgEnum("workflow_run_origin", [
    "manual",
    "api",
    "public-share",
    "public-template",
    "workspace",
]);

export const WorkflowRunStatusSchema = z.enum(workflowRunStatus.enumValues);

export const WorkflowRunOriginSchema = z.enum(workflowRunOrigin.enumValues);
export type WorkflowRunOriginType = z.infer<typeof WorkflowRunOriginSchema>;

export const machineGPUOptions = ["T4", "L4", "A10G", "A100", "A100-80GB", "H100"] as const;
export const machineGPUOptionsTypes = pgEnum("machine_gpu", machineGPUOptions);
export type machineGPUOptionsTypes = (typeof machineGPUOptions)[number];

export const workspace_machineGPUOptions = ["4090"] as const;
export const workspace_machineGPUOptionsTypes = pgEnum(
    "workspace_machine_gpu",
    workspace_machineGPUOptions,
);
export type workspace_machineGPUOptionsType =
    (typeof workspace_machineGPUOptions)[number];

export const machinesType = pgEnum("machine_type", [
    "classic",
    "runpod-serverless",
    "modal-serverless",
    "comfy-deploy-serverless",
    "workspace",
    "workspace-v2"
]);

export const machineStatusTypes = [
    "not-started",
    "ready",
    "building",
    "error",
    "running",
    "paused",
    "starting",
] as const;

export const machinesStatus = pgEnum("machine_status", machineStatusTypes);

export const machineBuilderVersionTypes = ["2", "3"] as const;
export const machineBuilderVersion = pgEnum(
    "machine_builder_version",
    machineBuilderVersionTypes,
);

// We still want to keep the workflow run record.
export const workflowRunsTable = dbSchema.table(
    "workflow_runs",
    {
        id: uuid("id").primaryKey().defaultRandom().notNull(),
        // when workflow version deleted, still want to keep this record
        workflow_version_id: uuid("workflow_version_id").references(
            () => workflowVersionTable.id,
            {
                onDelete: "set null",
            },
        ),
        workflow_inputs:
            jsonb("workflow_inputs").$type<Record<string, string | number>>(),
        workflow_id: uuid("workflow_id")
            .references(() => workflowTable.id, {
                onDelete: "cascade",
            }),
        // Use for overriding the workflow run
        workflow_api: jsonb("workflow_api").$type<z.infer<typeof workflowAPIType>>(),
        // when machine deleted, still want to keep this record
        machine_id: uuid("machine_id").references(() => machinesTable.id, {
            onDelete: "set null",
        }),
        origin: workflowRunOrigin("origin").notNull().default("api"),
        status: workflowRunStatus("status").notNull().default("not-started"),
        ended_at: timestamp("ended_at"),
        // comfy deploy run created time
        created_at: timestamp("created_at").defaultNow().notNull(),
        // last time that the run was updated
        updated_at: timestamp("updated_at").defaultNow().notNull(),
        // modal gpu cold start begin
        queued_at: timestamp("queued_at"),
        // modal gpu function actual start time
        started_at: timestamp("started_at"),
        gpu_event_id: text("gpu_event_id"),
        gpu: machineGPUOptionsTypes("gpu"),
        machine_version: text("machine_version"),
        machine_type: machinesType("machine_type"),
        modal_function_call_id: text("modal_function_call_id"),
        user_id: text("user_id"),
        org_id: text("org_id"),
        run_log: jsonb("run_log").$type<LogsType>(),
        live_status: text("live_status"),
        progress: real("progress").default(0).notNull(),
        is_realtime: boolean("is_realtime").default(false).notNull(),
        webhook: text("webhook"),
        webhook_status: webhook_status("webhook_status"),
        webhook_intermediate_status: boolean("webhook_intermediate_status").default(false).notNull(),
    },
    (table) => {
        return {
            idx_workflow_runs_workflow_id: index("idx_workflow_runs_workflow_id").on(
                table.workflow_id,
            ),
        };
    },
);

export const workflowRunRelations = relations(
    workflowRunsTable,
    ({ one, many }) => ({
        machine: one(machinesTable, {
            fields: [workflowRunsTable.machine_id],
            references: [machinesTable.id],
        }),
        version: one(workflowVersionTable, {
            fields: [workflowRunsTable.workflow_version_id],
            references: [workflowVersionTable.id],
        }),
        outputs: many(workflowRunOutputs),
        workflow: one(workflowTable, {
            fields: [workflowRunsTable.workflow_id],
            references: [workflowTable.id],
        }),
    }),
);

// We still want to keep the workflow run record.
export const workflowRunOutputs = dbSchema.table(
    "workflow_run_outputs",
    {
        id: uuid("id").primaryKey().defaultRandom().notNull(),
        run_id: uuid("run_id")
            .notNull()
            .references(() => workflowRunsTable.id, {
                onDelete: "cascade",
            }),
        data: jsonb("data").$type<any>(),
        node_meta: jsonb("node_meta").$type<any>(),

        created_at: timestamp("created_at").defaultNow().notNull(),
        updated_at: timestamp("updated_at").defaultNow().notNull(),
    },
    (table) => {
        return {
            idx_workflow_run_outputs_run_id: index(
                "idx_workflow_run_outputs_run_id",
            ).on(table.run_id),
        };
    },
);
export const workflowOutputRelations = relations(
    workflowRunOutputs,
    ({ one }) => ({
        run: one(workflowRunsTable, {
            fields: [workflowRunOutputs.run_id],
            references: [workflowRunsTable.id],
        }),
    }),
);

export const ExtraDockerCommandsType = z.array(
    z.object({
        commands: z.array(z.string()),
        when: z.union([z.literal("before"), z.literal("after")]),
    }),
);

// when user delete, also delete all the workflow versions
export const machinesTable = dbSchema.table(
    "machines",
    {
        id: uuid("id").primaryKey().defaultRandom().notNull(),
        user_id: text("user_id")
            .references(() => usersTable.id, {
                onDelete: "cascade",
            })
            .notNull(),
        name: text("name").notNull(),
        org_id: text("org_id"),
        endpoint: text("endpoint").notNull(),
        created_at: timestamp("created_at").defaultNow().notNull(),
        updated_at: timestamp("updated_at").defaultNow().notNull(),
        disabled: boolean("disabled").default(false).notNull(),
        auth_token: text("auth_token"),
        type: machinesType("type").notNull().default("classic"),
        status: machinesStatus("status").notNull().default("ready"),
        static_assets_status: machinesStatus("static_assets_status").notNull().default("not-started"),
        machine_version: text("machine_version"),
        machine_builder_version: machineBuilderVersion(
            "machine_builder_version",
        ).default("2"),
        snapshot: jsonb("snapshot").$type<any>(),
        models: jsonb("models").$type<any>(),
        gpu: machineGPUOptionsTypes("gpu"),
        ws_gpu: workspace_machineGPUOptionsTypes("ws_gpu"),
        pod_id: text("pod_id"),
        base_docker_image: text("base_docker_image"),
        allow_concurrent_inputs: integer("allow_concurrent_inputs").default(1),
        concurrency_limit: integer("concurrency_limit").default(2),
        legacy_mode: boolean("legacy_mode").default(false).notNull(),
        ws_timeout: integer("ws_timeout").default(2),
        run_timeout: integer("run_timeout")
            .default(60 * 5)
            .notNull(),
        idle_timeout: integer("idle_timeout").default(60).notNull(),
        build_machine_instance_id: text("build_machine_instance_id"),
        build_log: text("build_log"),
        modal_app_id: text("modal_app_id"),
        target_workflow_id: uuid("target_workflow_id").references(
            (): AnyPgColumn => workflowTable.id,
            {
                onDelete: "set null",
            },
        ),
        dependencies: jsonb("dependencies").$type<any>(),
        extra_docker_commands: jsonb("extra_docker_commands").$type<any>(),
        install_custom_node_with_gpu: boolean(
            "install_custom_node_with_gpu",
        ).default(false),
        deleted: boolean("deleted").default(false).notNull(),
        keep_warm: integer("keep_warm").default(0).notNull(),
        allow_background_volume_commits: boolean("allow_background_volume_commits").default(false).notNull(),
        gpu_workspace: boolean("gpu_workspace").default(false).notNull(),

        docker_command_steps: jsonb("docker_command_steps").$type<any>(),
        comfyui_version: text("comfyui_version"),

        python_version: text("python_version"),
        extra_args: text("extra_args"),
        prestart_command: text("prestart_command"),

        retrieve_static_assets: boolean("retrieve_static_assets").default(false),
        object_info: jsonb("object_info").$type<any>(),
        object_info_str: text('object_info_str').$type<any>(),
        filename_list_cache: jsonb("filename_list_cache").$type<any>(),
        extensions: jsonb("extensions").$type<any>(),
    },
    (table) => {
        return {
            // We might have to handle case with user_id and deleted
            idx_machines_org_id_deleted: index("idx_machines_org_id_deleted").on(
                table.org_id,
                table.deleted,
            ),
        };
    },
);

export const machinesRelations = relations(machinesTable, ({ one }) => ({
    target_workflow: one(workflowTable, {
        fields: [machinesTable.target_workflow_id],
        references: [workflowTable.id],
    }),
}));

export const snapshotType = z.object({
    comfyui: z.string(),
    git_custom_nodes: z.record(
        z.object({
            hash: z.string(),
            disabled: z.boolean(),
        }),
    ),
    file_custom_nodes: z.array(z.any()),
});

export const insertMachineSchema = createInsertSchema(machinesTable, {
    name: (schema) => schema.name.default("My Machine"),
    endpoint: (schema) => schema.endpoint.default("http://127.0.0.1:8188"),
    type: (schema) => schema.type.default("classic"),
});

export const showcaseMedia = z.array(
    z.object({
        url: z.string().url(),
        isCover: z.boolean().default(false),
    }),
).default([]);

export const OutputFileType = z.object({
    url: z.string(),
    filename: z.string(),
});


const runOutputs = z.array(
    z.object({
        data: z.object({
            images: z.array(OutputFileType).optional(),
            files: z.array(OutputFileType).optional(),
            gifs: z.array(OutputFileType).optional(),
            text: z.array(z.string()).optional(),
        }),
    }),
);

export const WebookRequestBody = z.object({
    status: WorkflowRunStatusSchema,
    live_status: z.string(),
    progress: z.number(),
    run_id: z.string(),
    outputs: runOutputs,
});

export const share_options = z.object({
    allowClone: z.boolean().default(true).optional(),
    showComfyUI: z.boolean().default(true).optional().describe("Show ComfyUI"),
})

export const showcaseMediaNullable = z
    .array(
        z.object({
            url: z.string(),
            isCover: z.boolean().default(false),
        }),
    )
    .nullable();

export const deploymentsTable = dbSchema.table("deployments", {
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    user_id: text("user_id")
        .references(() => usersTable.id, {
            onDelete: "cascade",
        })
        .notNull(),
    org_id: text("org_id"),
    workflow_version_id: uuid("workflow_version_id")
        .notNull()
        .references(() => workflowVersionTable.id),
    workflow_id: uuid("workflow_id")
        .notNull()
        .references(() => workflowTable.id, {
            onDelete: "cascade",
        }),
    machine_id: uuid("machine_id")
        .notNull()
        .references(() => machinesTable.id),
    share_slug: text("share_slug").unique(),
    description: text("description"),
    share_options:
        jsonb("share_options").$type<z.infer<typeof share_options>>(),
    showcase_media:
        jsonb("showcase_media").$type<z.infer<typeof showcaseMedia>>(),
    environment: deploymentEnvironment("environment").notNull(),
    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
});

export const shareSettings = createSelectSchema(deploymentsTable, {
    // id: z.string(),
    share_options: share_options.partial().describe("Options")
}).pick({
    id: true,
    showcase_media: true,
    description: true,
    share_options: true,
});

export const publicShareDeployment = z.object({
    id: z.string(),
    description: z.string().nullable(),
    showcase_media: showcaseMedia.describe("Showcase media").optional(),
});

// createInsertSchema(deploymentsTable, {
//   description: (schema) => schema.description.default(""),
//   showcase_media: () => showcaseMedia.default([]),
// }).pick({
//   description: true,
//   showcase_media: true,
// });

export const deploymentsRelations = relations(deploymentsTable, ({ one }) => ({
    machine: one(machinesTable, {
        fields: [deploymentsTable.machine_id],
        references: [machinesTable.id],
    }),
    version: one(workflowVersionTable, {
        fields: [deploymentsTable.workflow_version_id],
        references: [workflowVersionTable.id],
    }),
    workflow: one(workflowTable, {
        fields: [deploymentsTable.workflow_id],
        references: [workflowTable.id],
    }),
    user: one(usersTable, {
        fields: [deploymentsTable.user_id],
        references: [usersTable.id],
    }),
}));

export const apiKeyTable = dbSchema.table("api_keys", {
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    key: text("key").notNull().unique(),
    name: text("name").notNull(),
    user_id: text("user_id")
        .references(() => usersTable.id, {
            onDelete: "cascade",
        })
        .notNull(),
    org_id: text("org_id"),
    revoked: boolean("revoked").default(false).notNull(),
    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
});

export const userUsageTable = dbSchema.table("user_usage", {
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    org_id: text("org_id"),
    user_id: text("user_id")
        .references(() => usersTable.id, {
            onDelete: "cascade",
        })
        .notNull(),
    usage_time: real("usage_time").default(0).notNull(),
    created_at: timestamp("created_at").defaultNow().notNull(),
    ended_at: timestamp("ended_at").defaultNow().notNull(),
});

export const authRequestsTable = dbSchema.table("auth_requests", {
    request_id: text("request_id").primaryKey().notNull(),
    user_id: text("user_id"),
    org_id: text("org_id"),
    api_hash: text("api_hash"),
    created_at: timestamp("created_at").defaultNow().notNull(),
    expired_date: timestamp("expired_date"),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
});

export const resourceUpload = pgEnum("resource_upload", [
    "started",
    "success",
    "failed",
    "cancelled",
]);

export const modelUploadType = pgEnum("model_upload_type", [
    "civitai",
    "download-url",
    "huggingface", // remove?
    "other",
]);

// https://www.answeroverflow.com/m/1125106227387584552
export const modelTypes = [
    "checkpoint",
    "lora",
    "embedding",
    "vae",
    "clip",
    "clip_vision",
    "configs",
    "controlnet",
    "upscale_models",
    "ipadapter",
    "gligen",
    "unet",
    "custom",
    "custom_node",
] as const;
export const modelType = pgEnum("model_type", modelTypes);
export type modelEnumType = (typeof modelTypes)[number];

export const modelTable = dbSchema.table("models", {
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    user_id: text("user_id").references(() => usersTable.id, {}), // perhaps a "special" user_id for global models
    org_id: text("org_id"),
    description: text("description"),

    user_volume_id: uuid("user_volume_id")
        .notNull()
        .references(() => userVolume.id, {
            onDelete: "cascade",
        }),

    model_name: text("model_name"),
    folder_path: text("folder_path"), // folder-path in the volume
    target_symlink_path: text("target_symlink_path"),

    civitai_id: text("civitai_id"),
    civitai_version_id: text("civitai_version_id"),
    civitai_url: text("civitai_url"),
    civitai_download_url: text("civitai_download_url"),
    civitai_model_response: jsonb("civitai_model_response").$type<
        z.infer<typeof CivitaiModelResponse>
    >(),

    // for our own storage
    hf_url: text("hf_url"),
    s3_url: text("s3_url"),
    download_progress: integer("download_progress").default(0),

    user_url: text("client_url"),
    filehash_sha256: text("file_hash_sha256"),

    is_public: boolean("is_public").notNull().default(true),
    status: resourceUpload("status").notNull().default("started"),
    upload_machine_id: text("upload_machine_id"), // TODO: review if actually needed
    upload_type: modelUploadType("upload_type").notNull(),
    model_type: modelType("model_type").default("checkpoint"),
    error_log: text("error_log"),

    deleted: boolean("deleted").default(false).notNull(),
    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
});

export const userVolume = dbSchema.table("user_volume", {
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    user_id: text("user_id").references(() => usersTable.id, {
        // onDelete: "cascade",
    }),
    org_id: text("org_id"),
    volume_name: text("volume_name").notNull(),
    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
    disabled: boolean("disabled").default(false).notNull(),
});

export const modelRelations = relations(modelTable, ({ one }) => ({
    user: one(usersTable, {
        fields: [modelTable.user_id],
        references: [usersTable.id],
    }),
    volume: one(userVolume, {
        fields: [modelTable.user_volume_id],
        references: [userVolume.id],
    }),
}));

export const modalVolumeRelations = relations(userVolume, ({ many, one }) => ({
    model: many(modelTable),
    user: one(usersTable, {
        fields: [userVolume.user_id],
        references: [usersTable.id],
    }),
}));

const subscriptionPlanValues = [
    "basic",
    "pro",
    "enterprise",
    "business",
    "ws_basic",
    "ws_pro",
] as const;
export const subscriptionPlan = pgEnum("subscription_plan", subscriptionPlanValues);
export type SubscriptionPlanType = (typeof subscriptionPlanValues)[number];

export const subscriptionPlanStatus = pgEnum("subscription_plan_status", [
    "active",
    "deleted",
    "paused",
]);

export const subscriptionStatusTable = dbSchema.table("subscription_status", {
    stripe_customer_id: text("stripe_customer_id").primaryKey().notNull(),
    user_id: text("user_id"),
    org_id: text("org_id"),
    plan: subscriptionPlan("plan").notNull(),
    status: subscriptionPlanStatus("status").notNull(),
    subscription_id: text("subscription_id"),
    subscription_item_plan_id: text("subscription_item_plan_id"),
    subscription_item_api_id: text("subscription_item_api_id"), // not used in new version of billing
    cancel_at_period_end: boolean("cancel_at_period_end").default(false),
    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
    trial_end: integer("trial_end"),
    trial_start: integer("trial_start"),
    // last_invoice_date: timestamp("last_invoice_date").defaultNow().notNull(),
    last_invoice_timestamp: timestamp("last_invoice_timestamp").defaultNow().notNull(),
});

export const credits = dbSchema.table("credits", {
    user_or_org_id: text("user_or_org_id").primaryKey().notNull(),
    ws_credit: real("ws_credit").default(100).notNull(),
    last_updated: timestamp("last_updated").defaultNow().notNull(),
});

export const gpuProviders = ["modal", "runpod"] as const;
export const gpuProviderType = pgEnum("gpu_provider", gpuProviders);
export type gpuProviderType = (typeof gpuProviders)[number];

export const gpuEvents = dbSchema.table("gpu_events", {
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    user_id: text("user_id")
        .references(() => usersTable.id)
        .notNull(),
    org_id: text("org_id"),
    machine_id: uuid("machine_id").references(() => machinesTable.id, {
        onDelete: "set null",
    }),
    start_time: timestamp("start_time"),
    end_time: timestamp("end_time"),
    gpu: machineGPUOptionsTypes("gpu"),
    ws_gpu: workspace_machineGPUOptionsTypes("ws_gpu"),
    providerType: gpuProviderType("gpu_provider").notNull(),
    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
});

const trainingTypeEnum = ["flux-lora"] as const;
export const trainingType = pgEnum("training_type", trainingTypeEnum);

export const trainings = dbSchema.table("trainings", {
    id: text("id").primaryKey().notNull(),
    user_id: text("user_id")
        .references(() => usersTable.id)
        .notNull(),
    org_id: text("org_id"),

    name: text("name").notNull(),
    inputs: jsonb("inputs"),
    outputs: jsonb("outputs"),
    status: resourceUpload("status").notNull().default("started"),
    type: trainingType("type").notNull(),

    request_id: text("request_id"),

    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
});

export const formSubmissionsTable = dbSchema.table("form_submissions", {
    id: uuid("id").primaryKey().defaultRandom().notNull(),
    user_id: text("user_id").references(() => usersTable.id, {
        onDelete: "set null",
    }),
    org_id: text("org_id"),
    inputs: jsonb("inputs").$type<Record<string, any>>(),
    call_booked: boolean("call_booked").default(false).notNull(),
    discord_thread_id: text("discord_thread_id"),
    created_at: timestamp("created_at").defaultNow().notNull(),
    updated_at: timestamp("updated_at").defaultNow().notNull(),
});

export type FormSubmissionType = InferSelectModel<typeof formSubmissionsTable>;

export type UserType = InferSelectModel<typeof usersTable>;
export type WorkflowType = InferSelectModel<typeof workflowTable>;
export type WorkflowRunType = InferSelectModel<typeof workflowRunsTable>;
export type WorkflowRunOutputType = InferSelectModel<typeof workflowRunOutputs>;
export type MachineType = InferSelectModel<typeof machinesTable>;
export type WorkflowVersionType = InferSelectModel<typeof workflowVersionTable>;
export type DeploymentType = InferSelectModel<typeof deploymentsTable>;
export type ModelType = InferSelectModel<typeof modelTable>;
export type UserVolumeType = InferSelectModel<typeof userVolume>;
export type UserUsageType = InferSelectModel<typeof userUsageTable>;
export type APIKeyType = InferSelectModel<typeof apiKeyTable>;
export type TrainingType = InferSelectModel<typeof trainings>;
export type SubscriptionStatusType = InferSelectModel<
    typeof subscriptionStatusTable
>;
export type GpuEventType = InferSelectModel<typeof gpuEvents>;
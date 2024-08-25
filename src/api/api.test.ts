import { expect, test, describe } from "bun:test";

const API_URL = "http://localhost:3011"; // Adjust this to your API's URL

const token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidXNlcl8yWkE2dnVLRDNJSlhqdTE2b0pWUUdMQmNXd2ciLCJvcmdfaWQiOiJvcmdfMmJXUTFGb1dDM1dybzM5MVR1cmtlVkc3N3BDIiwiaWF0IjoxNzIwMTM5NjMyfQ.BBSt5kWJRPDAwNx2Sk_EJU5XRoUJpmTv1hLZvuf1r-M"
const run_id = "01babcd2-52a8-4fb6-82b3-949a4cdc94d8"
const workflow_id = "8fb57e53-f899-4c04-bd7c-30c22dc0d841"
const deployment_id = "49a5d198-542a-4791-a84a-6e66a2f3182c"

describe("API Tests", () => {
    test("GET /api/run without token should return 401", async () => {
        const response = await fetch(`${API_URL}/api/run?run_id=${run_id}`);
        expect(response.status).toBe(401);
        const data = await response.json();
        expect(data).toHaveProperty("detail");
        expect(data.detail).toBe("Invalid or missing token");
    });

    test("GET /api/run should return 200", async () => {
        const response = await fetch(`${API_URL}/api/run?&run_id=${run_id}`,
            {
                headers: {
                    "Authorization": `Bearer ${token}`
                }
            }
        );
        expect(response.status).toBe(200);
        const data = await response.json();
    });

    test("run endpoint with a wrong token", async () => {
        const response = await fetch(`${API_URL}/api/run?&run_id=02327d80-6edd-4a19-8611-c48f0abe4006`,
            {
                headers: {
                    "Authorization": `Bearer ${token}`
                }
            }
        );
        expect(response.status).toBe(404);
        const data = await response.json();
        expect(data.detail).toBe("Run not found");
    });

    test("run endpoint with revoked token", async () => {
        const revoked_token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJ1c2VyX2lkIjoidXNlcl8yWkE2dnVLRDNJSlhqdTE2b0pWUUdMQmNXd2ciLCJvcmdfaWQiOiJvcmdfMmJXUTFGb1dDM1dybzM5MVR1cmtlVkc3N3BDIiwiaWF0IjoxNzI0MzYyNzE3fQ.lcIiYjxngXwMhPTxHozYazzW-jso_-QPNCsL0fJw24g";

        const req = await fetch(`${API_URL}/api/run?run_id=${run_id}`, {
            headers: { "Authorization": `Bearer ${revoked_token}` }
        });

        expect(req.status).toBe(401);
        const data = await req.json();
        expect(data).toHaveProperty("detail");
        expect(data.detail).toBe("Revoked token");
    });
});

describe("POST /api/run", () => {
    // test("POST /api/run should return 200", async () => {
    //     const response = await fetch(`${API_URL}/api/run`,
    //         {
    //             method: "POST",
    //             headers: {
    //                 "Authorization": `Bearer ${token}`
    //             },
    //             body: JSON.stringify({
    //                 workflow_id: workflow_id,
    //                 inputs: {
    //                     "input1": "value1"
    //                 }
    //             })
    //         }
    //     );
    // });

    test("POST /api/run should return 200", async () => {
        const response = await fetch(`${API_URL}/api/run`,
            {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`
                },
                body: JSON.stringify({
                    deployment_id: deployment_id,
                    inputs: {
                        "input1": "value1"
                    }
                })
            }
        );

        expect(response.status).toBe(404);
        const data = await response.json();
        expect(data).toHaveProperty("detail");
        expect(data.detail).toBe("Deployment not found");
    });

    test("POST /api/run should return 200", async () => {
        const response = await fetch(`${API_URL}/api/run`,
            {
                method: "POST",
                headers: {
                    "Authorization": `Bearer ${token}`
                },
                body: JSON.stringify({
                    deployment_id: "0d4e1bd3-9c35-45d4-882d-ae008c7fc9e3",
                    inputs: {
                        "input1": "value1"
                    }
                })
            }
        );

        expect(response.status).toBe(200);
        const data = await response.json();
    });
});

describe("File upload", () => {
    test("POST /api/run should return 200", async () => {
        const imageUrl = 'https://comfy-deploy-output.s3.amazonaws.com/outputs/runs/e129217b-c386-48e4-a0ab-5ab53bc2f66d/ComfyUI_00009_.png';
        const imageReponse = await fetch(imageUrl);
        const arrayBuffer = await imageReponse.arrayBuffer();
        const blob = new Blob([arrayBuffer], { type: 'image/png' });

        const response = await fetch(`${API_URL}/api/file-upload?file_name=test&run_id=11df9525-e089-41a3-94a0-3109766e39f2&type=image/png`,
            {
                method: "GET",
                headers: {
                    "Authorization": `Bearer ${token}`
                }
            }
        );

        expect(response.status).toBe(200);
        const data = await response.json();
        expect(data).toHaveProperty("url");
        expect(data).toHaveProperty("download_url");

        console.log(data);

        const upload_response = await fetch(data.url, {
            method: "PUT",
            body: blob,
            headers: {
                'Content-Type': 'image/png',
                'x-amz-acl': 'public-read'  // Add this line
            }
        });

        if (!upload_response.ok) {
            const errorText = await upload_response.text();
            console.error(`Upload failed with status ${upload_response.status}:`, errorText);
        }

        expect(upload_response.status).toBe(200);
    });
})
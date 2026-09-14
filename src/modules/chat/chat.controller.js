import crypto from 'crypto';
import { chatPipeline } from './services/rag/rag-system.service.js';
import { redisGet, redisSet } from '../core/services/infrastructure/redis.service.js';
import { redisClient } from '../../config/redis.js';
import {
  createChatMessage,
  deleteChatMessage,
  deleteProjectChatMessages,
  getChatMessageById,
  getProjectChatMessages,
  updateChatMessage
} from './services/rag/projectChatMessage.service.js';
import logger from '../../utils/logger.js';

const CHAT_CACHE_PREFIX = 'cache:chat:v3';
const CHAT_CACHE_TTL_SECONDS = process.env.CHAT_CACHE_TTL_SECONDS
  ? Number(process.env.CHAT_CACHE_TTL_SECONDS)
  : undefined;

const getActiveChatModel = () => {
  if (process.env.AI_PROVIDER === 'gemini') return process.env.AI_MODEL || 'gemini';
  if (process.env.AI_PROVIDER === 'openai') return process.env.AI_MODEL || 'openai';
  return process.env.AI_MODEL || process.env.OLLAMA_MODEL || process.env.RAG_MODEL || 'ollama';
};

const normalizeQuestionForCache = (question) => String(question || '')
  .trim()
  .toLowerCase()
  .replace(/\s+/g, ' ');

const buildChatCacheKey = (projectId, question) => {
  const normalizedQuestion = normalizeQuestionForCache(question);
  const questionHash = crypto
    .createHash('sha256')
    .update(normalizedQuestion)
    .digest('hex');

  return `${CHAT_CACHE_PREFIX}:project:${projectId}:question:${questionHash}`;
};

const saveChatMessageSafely = async (payload) => {
  try {
    return await createChatMessage(payload);
  } catch (error) {
    logger.warn('[CHAT HISTORY] Không thể lưu lịch sử chat:', error?.message || error);
    return null;
  }
};

const parseProjectId = (value) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
};

const parseMessageId = (value) => {
  const parsed = Number(value);
  return Number.isInteger(parsed) && parsed > 0 ? parsed : null;
};

const isValidUuid = (value) => {
  return typeof value === 'string' && /^[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i.test(value);
};

/**
 * Gọi API tới hệ thống RAG Python (Rag_System_AI).
 */
export const callRagSystemService = async ({ query, projectId, userId, topK = 5, model, temperature }) => {
  const rawBaseUrl = process.env.RAG_SERVICE_URL || 'http://127.0.0.1:8001';
  const baseUrl = rawBaseUrl.replace(/\/+$/, '');
  const ragUrl = `${baseUrl}/api/v1/chat`;
  logger.info(`[CHAT RAG AI] Đang gửi yêu cầu tới Rag_System_AI: ${ragUrl}`);

  const controller = new AbortController();
  const timeoutMs = Number(process.env.AI_TIMEOUT_MS) || 120000;
  const timeoutId = setTimeout(() => controller.abort(), timeoutMs);

  try {
    const response = await fetch(ragUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        query,
        project_id: projectId ? Number(projectId) : undefined,
        user_id: userId ? String(userId) : undefined,
        top_k: topK,
        model: model || undefined,
        temperature: temperature || undefined,
        save_history: false, // Quản lý lưu history tập trung tại Trending BE để đồng bộ DB
      }),
      signal: controller.signal,
    });

    clearTimeout(timeoutId);

    if (!response.ok) {
      const errorText = await response.text();
      throw new Error(`Rag_System_AI HTTP error ${response.status}: ${errorText}`);
    }

    const data = await response.json();
    return {
      answer: data.answer || '',
      model: data.model || 'rag-system-ai',
      citations: data.citations || [],
      contexts: data.contexts || [],
      latencyBreakdown: data.latency_breakdown || {},
      tokens: {
        promptTokens: 0,
        completionTokens: 0,
        totalTokens: 0,
      },
    };
  } catch (error) {
    clearTimeout(timeoutId);
    if (error.name === 'AbortError') {
      throw new Error(`Hết thời gian kết nối tới Rag_System_AI (${timeoutMs}ms)`);
    }
    throw error;
  }
};

export const chatRagSystem = async (request, reply) => {
  const requestBody = request.body;
  const projectId = parseProjectId(requestBody ? (requestBody.project_id || requestBody.projectId) : null);
  const rawUserId = request.user?.user_id || requestBody?.user_id || null;
  const userId = isValidUuid(rawUserId) ? rawUserId : null;
  const startedAt = Date.now();
  const userQuestion = requestBody ? (requestBody.message || requestBody.query) : null;

  if (!requestBody || !userQuestion) {
    return reply.status(400).send({
      success: false,
      message: 'Message (hoặc query) là bắt buộc.'
    });
  }

  if (!projectId) {
    return reply.status(400).send({
      success: false,
      message: 'project_id (hoặc projectId) là bắt buộc.'
    });
  }

  let userMessage = null;

  try {
    const cacheKey = buildChatCacheKey(projectId, userQuestion);

    logger.info(`[CHAT API] Nhận yêu cầu: ${userQuestion} (Project ID: ${projectId}, User ID: ${userId})`);

    userMessage = await saveChatMessageSafely({
      projectId,
      userId,
      role: 'USER',
      content: userQuestion,
      status: 'COMPLETED'
    });

    if (redisClient?.isOpen) {
      try {
        const cachedResult = await redisGet(cacheKey);
        if (cachedResult) {
          logger.info(`[CHAT CACHE] Cache hit cho Project ID: ${projectId}, Key: ${cacheKey}`);
          const parsedResult = JSON.parse(cachedResult);
          const assistantMessage = await saveChatMessageSafely({
            projectId,
            userId,
            role: 'ASSISTANT',
            content: parsedResult.answer,
            model: 'cache',
            latencyMs: Date.now() - startedAt,
            status: 'COMPLETED'
          });

          return reply.status(200).send({
            success: true,
            answer: parsedResult.answer,
            model: 'cache',
            citations: parsedResult.citations || [],
            contexts: parsedResult.contexts || [],
            table: parsedResult.table || null,
            messages: {
              user_message_id: userMessage?.message_id || null,
              assistant_message_id: assistantMessage?.message_id || null
            }
          });
        }
        logger.info(`[CHAT CACHE] Cache miss cho Project ID: ${projectId}, Key: ${cacheKey}`);
      } catch (cacheError) {
        logger.warn('[CHAT CACHE] Không thể đọc cache Redis, tiếp tục xử lý pipeline:', cacheError?.message || cacheError);
      }
    }

    let result = null;
    let fromFallback = false;
    let modelName = 'rag-system-ai';

    try {
      // 1. Ưu tiên kết nối trực tiếp tới Rag_System_AI (FastAPI service)
      result = await callRagSystemService({
        query: userQuestion,
        projectId,
        userId,
        topK: requestBody.top_k || requestBody.topK || 5,
        model: requestBody.model,
        temperature: requestBody.temperature
      });
      modelName = result.model || modelName;
      logger.info(`[CHAT RAG AI] Xử lý thành công từ Rag_System_AI (Model: ${modelName})`);
    } catch (ragServiceError) {
      logger.warn(`[CHAT RAG AI] Rag_System_AI không phản hồi hoặc xảy ra lỗi: ${ragServiceError.message}. Chuyển sang Fallback Pipeline nội bộ...`);
      // 2. Dự phòng bằng chatPipeline nội bộ
      result = await chatPipeline(userQuestion, projectId, userId);
      fromFallback = result.fromFallback ?? true;
      modelName = getActiveChatModel();
    }

    if (redisClient?.isOpen && !fromFallback) {
      try {
        await redisSet(cacheKey, JSON.stringify(result), CHAT_CACHE_TTL_SECONDS);
        logger.info(`[CHAT CACHE] Đã lưu cache cho Project ID: ${projectId}, Key: ${cacheKey}`);
      } catch (cacheError) {
        logger.warn('[CHAT CACHE] Không thể lưu cache Redis:', cacheError?.message || cacheError);
      }
    } else if (fromFallback) {
      logger.info('[CHAT CACHE] Bỏ qua lưu cache vì AI dùng Fallback (kết quả có thể chưa tối ưu).');
    }

    const assistantMessage = await saveChatMessageSafely({
      projectId,
      userId,
      role: 'ASSISTANT',
      content: result.answer,
      model: modelName,
      promptTokens: result.tokens?.promptTokens || 0,
      completionTokens: result.tokens?.completionTokens || 0,
      totalTokens: result.tokens?.totalTokens || 0,
      latencyMs: Date.now() - startedAt,
      status: fromFallback ? 'ERROR' : 'COMPLETED'
    });

    return reply.status(200).send({
      success: true,
      answer: result.answer,
      model: modelName,
      citations: result.citations || [],
      contexts: result.contexts || [],
      table: result.table || null,
      messages: {
        user_message_id: userMessage?.message_id || null,
        assistant_message_id: assistantMessage?.message_id || null
      }
    });
  } catch (error) {
    logger.error('[CHAT API] Lỗi xử lý yêu cầu Chatbot:', error);

    await saveChatMessageSafely({
      projectId,
      userId,
      role: 'ASSISTANT',
      content: 'Đã xảy ra lỗi hệ thống, vui lòng thử lại sau.',
      model: getActiveChatModel(),
      latencyMs: Date.now() - startedAt,
      status: 'ERROR'
    });

    return reply.status(500).send({
      success: false,
      message: 'Đã xảy ra lỗi hệ thống, vui lòng thử lại sau.'
    });
  }
};

export const createChatMessageHandler = async (request, reply) => {
  try {
    const projectId = parseProjectId(request.params.projectId);
    const userId = request.user?.user_id;

    if (!projectId) {
      return reply.status(400).send({ success: false, message: 'projectId is invalid' });
    }

    const message = await createChatMessage({
      projectId,
      userId,
      role: request.body.role,
      content: request.body.content,
      model: request.body.model,
      promptTokens: request.body.prompt_tokens ?? request.body.promptTokens,
      completionTokens: request.body.completion_tokens ?? request.body.completionTokens,
      totalTokens: request.body.total_tokens ?? request.body.totalTokens,
      latencyMs: request.body.latency_ms ?? request.body.latencyMs,
      status: request.body.status || 'COMPLETED'
    });

    return reply.status(201).send({ success: true, data: message });
  } catch (error) {
    logger.error('[CHAT MESSAGE] Lỗi tạo message:', error);
    return reply.status(500).send({ success: false, message: 'Đã xảy ra lỗi khi tạo chat message.' });
  }
};

export const getChatHistory = async (request, reply) => {
  try {
    const projectId = parseProjectId(request.params.projectId);
    const userId = request.user?.user_id;

    if (!projectId) {
      return reply.status(400).send({ success: false, message: 'projectId is invalid' });
    }

    const messages = await getProjectChatMessages(projectId, userId, {
      limit: request.query.limit,
      offset: request.query.offset,
      order: request.query.order
    });

    return reply.status(200).send({ success: true, data: messages });
  } catch (error) {
    logger.error('[CHAT MESSAGE] Lỗi lấy lịch sử chat:', error);
    return reply.status(500).send({ success: false, message: 'Đã xảy ra lỗi khi lấy lịch sử chat.' });
  }
};

export const getChatMessageDetail = async (request, reply) => {
  try {
    const projectId = parseProjectId(request.params.projectId);
    const messageId = parseMessageId(request.params.messageId);
    const userId = request.user?.user_id;

    if (!projectId || !messageId) {
      return reply.status(400).send({ success: false, message: 'projectId or messageId is invalid' });
    }

    const message = await getChatMessageById(messageId, projectId, userId);
    if (!message) {
      return reply.status(404).send({ success: false, message: 'Chat message not found' });
    }

    return reply.status(200).send({ success: true, data: message });
  } catch (error) {
    logger.error('[CHAT MESSAGE] Lỗi lấy chi tiết message:', error);
    return reply.status(500).send({ success: false, message: 'Đã xảy ra lỗi khi lấy chi tiết chat message.' });
  }
};

export const updateChatMessageHandler = async (request, reply) => {
  try {
    const projectId = parseProjectId(request.params.projectId);
    const messageId = parseMessageId(request.params.messageId);
    const userId = request.user?.user_id;

    if (!projectId || !messageId) {
      return reply.status(400).send({ success: false, message: 'projectId or messageId is invalid' });
    }

    const message = await updateChatMessage(messageId, projectId, userId, request.body);
    if (!message) {
      return reply.status(404).send({ success: false, message: 'Chat message not found' });
    }

    return reply.status(200).send({ success: true, data: message });
  } catch (error) {
    logger.error('[CHAT MESSAGE] Lỗi cập nhật message:', error);
    return reply.status(500).send({ success: false, message: 'Đã xảy ra lỗi khi cập nhật chat message.' });
  }
};

export const deleteChatMessageHandler = async (request, reply) => {
  try {
    const projectId = parseProjectId(request.params.projectId);
    const messageId = parseMessageId(request.params.messageId);
    const userId = request.user?.user_id;

    if (!projectId || !messageId) {
      return reply.status(400).send({ success: false, message: 'projectId or messageId is invalid' });
    }

    const result = await deleteChatMessage(messageId, projectId, userId);
    return reply.status(200).send({ success: true, ...result });
  } catch (error) {
    logger.error('[CHAT MESSAGE] Lỗi xóa một message:', error);
    return reply.status(500).send({ success: false, message: 'Đã xảy ra lỗi khi xóa chat message.' });
  }
};

export const clearChatHistory = async (request, reply) => {
  try {
    const projectId = parseProjectId(request.params.projectId);
    const userId = request.user?.user_id;

    if (!projectId) {
      return reply.status(400).send({ success: false, message: 'projectId is invalid' });
    }

    const result = await deleteProjectChatMessages(projectId, userId);
    return reply.status(200).send({ success: true, ...result });
  } catch (error) {
    logger.error('[CHAT MESSAGE] Lỗi xóa lịch sử chat:', error);
    return reply.status(500).send({ success: false, message: 'Đã xảy ra lỗi khi xóa lịch sử chat.' });
  }
};
